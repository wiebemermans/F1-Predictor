from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from f1_predictor.models.baselines import (
    ChampionshipPointsBaseline,
    QualifyingPositionBaseline,
    RacePredictor,
)
from f1_predictor.models.boosting import (
    XGBNDCGRankerPredictor,
    XGBRankerPredictor,
    XGBRegressorPredictor,
)
from f1_predictor.models.linear import RidgeFeaturePredictor
from f1_predictor.models.train import MODEL_FAMILIES

POINTS = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0}
FEATURES = ["start_position", "noise"]
LABEL_COLUMNS = ["finish_position", "position_text", "status", "points", "laps", "sprint_points"]
XGB_PREDICTORS = [XGBRegressorPredictor, XGBRankerPredictor, XGBNDCGRankerPredictor]
# Klein en snel genoeg voor tests, maar diep genoeg om zes posities te scheiden.
FAST = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3}


def _table(n_races: int = 8, n_drivers: int = 6, seed: int = 0) -> pd.DataFrame:
    """Synthetische featuretabel waarin de finish exact de startpositie volgt.

    Met kwalificatiepositie en punten, zodat ook de baselines erop kunnen draaien.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for number in range(n_races):
        start = rng.permutation(n_drivers) + 1
        frames.append(
            pd.DataFrame(
                {
                    "season": 2020 + number // 4,
                    "round": number % 4 + 1,
                    "race_name": f"Race {number}",
                    "driver_id": [f"d{index}" for index in range(n_drivers)],
                    "quali_position": pd.array(start, dtype="Int64"),
                    "start_position": start.astype(float),
                    "noise": rng.normal(size=n_drivers),
                    "finish_position": start,
                    "position_text": [str(position) for position in start],
                    "status": "Finished",
                    "points": [POINTS.get(position, 0.0) for position in start],
                    "laps": 50,
                    "sprint_points": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    last = data["race_name"] == data["race_name"].iloc[-1]
    return data[~last], data[last]


# --- Contract voor elke RacePredictor ---


def _model_factory(family: str) -> Callable[[], RacePredictor]:
    params = {} if family == "ridge" else FAST
    return partial(MODEL_FAMILIES[family], FEATURES, **params)


# Beide baselines plus elke familie uit MODEL_FAMILIES: een nieuw model valt automatisch onder
# dit contract (zie `RacePredictor` in models/baselines.py).
ALL_PREDICTORS: dict[str, Callable[[], RacePredictor]] = {
    "qualifying_position": QualifyingPositionBaseline,
    "championship_points": ChampionshipPointsBaseline,
    **{family: _model_factory(family) for family in MODEL_FAMILIES},
}


@pytest.mark.parametrize("make_predictor", ALL_PREDICTORS.values(), ids=list(ALL_PREDICTORS))
def test_every_predictor_returns_unique_ranks_without_race_labels(make_predictor):
    train, race = _split(_table())
    race = race.set_index(pd.Index([30, 10, 20, 50, 40, 60]))
    train_before, race_before = train.copy(), race.copy()

    predictor = make_predictor().fit(train)
    with_labels = predictor.predict(race)
    # De uitslag van de voorspelde race is een label: zonder die kolommen dezelfde voorspelling.
    without_labels = predictor.predict(race.drop(columns=LABEL_COLUMNS))

    assert with_labels.index.equals(race.index)
    assert sorted(with_labels.tolist()) == list(range(1, len(race) + 1))
    pd.testing.assert_series_equal(with_labels, without_labels)
    # `fit` en `predict` passen hun invoer niet aan.
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(race, race_before)


# --- XGBoost-voorspellers ---


@pytest.mark.parametrize("predictor_class", [XGBRegressorPredictor, XGBRankerPredictor])
def test_xgboost_predictors_follow_strong_signal(predictor_class):
    train, race = _split(_table())

    predicted = predictor_class(FEATURES, colsample_bytree=1.0, **FAST).fit(train).predict(race)

    assert predicted.index.equals(race.index)
    assert predicted.tolist() == race["finish_position"].tolist()


def test_ndcg_ranker_gets_top_of_field_right():
    # NDCG weegt de achterhoede nauwelijks (gain 2^label − 1), dus alleen de top moet kloppen.
    train, race = _split(_table())

    predicted = (
        XGBNDCGRankerPredictor(FEATURES, colsample_bytree=1.0, **FAST).fit(train).predict(race)
    )

    top = race["finish_position"] <= 3
    assert predicted[top].tolist() == race.loc[top, "finish_position"].tolist()


@pytest.mark.parametrize("predictor_class", XGB_PREDICTORS)
def test_xgboost_predictors_are_deterministic(predictor_class):
    train, race = _split(_table())

    first = predictor_class(FEATURES, **FAST).fit(train)._race_scores(race)
    second = predictor_class(FEATURES, **FAST).fit(train)._race_scores(race)

    np.testing.assert_array_equal(first, second)


def test_ranker_ignores_order_of_training_rows():
    train, race = _split(_table())
    shuffled = train.sample(frac=1.0, random_state=1)

    original = XGBRankerPredictor(FEATURES, **FAST).fit(train)._race_scores(race)
    reordered = XGBRankerPredictor(FEATURES, **FAST).fit(shuffled)._race_scores(race)

    np.testing.assert_array_equal(original, reordered)


def test_ranker_trains_on_reversed_finish_grouped_per_race(monkeypatch):
    captured = {}

    def fake_fit(self, features, label, qid=None, **kwargs):
        captured.update(features=features, label=np.asarray(label), qid=np.asarray(qid))
        return self

    monkeypatch.setattr(xgb.XGBRanker, "fit", fake_fit)
    train = pd.DataFrame(
        {
            "season": [2021, 2020, 2021, 2020, 2020],
            "round": [1, 2, 1, 2, 2],
            "driver_id": ["b", "c", "a", "a", "b"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "finish_position": [1, 3, 2, 1, 2],
        }
    )

    XGBRankerPredictor(["x"]).fit(train)

    # Gesorteerd per race (2020-2 vóór 2021-1) en binnen een race op driver_id.
    assert captured["qid"].tolist() == [0, 0, 0, 1, 1]
    assert captured["features"]["x"].tolist() == [4.0, 5.0, 2.0, 3.0, 1.0]
    assert captured["label"].tolist() == [2, 1, 0, 0, 1]


@pytest.mark.parametrize("predictor_class", XGB_PREDICTORS)
def test_xgboost_predictors_accept_missing_feature_values(predictor_class):
    data = _table()
    data.loc[data.index % 4 == 0, "noise"] = np.nan
    data.loc[data.index[-1], "start_position"] = np.nan
    train, race = _split(data)

    predicted = predictor_class(FEATURES, **FAST).fit(train).predict(race)

    assert sorted(predicted.tolist()) == list(range(1, len(race) + 1))


@pytest.mark.parametrize("predictor_class", XGB_PREDICTORS)
def test_xgboost_predictors_validate_input(predictor_class):
    train, race = _split(_table())

    with pytest.raises(RuntimeError, match="fit"):
        predictor_class(FEATURES).predict(race)
    with pytest.raises(ValueError, match="featurekolommen"):
        predictor_class(["missing"], **FAST).fit(train)
    with pytest.raises(ValueError, match="minstens één"):
        predictor_class([])
    with pytest.raises(ValueError, match="uniek"):
        predictor_class(["noise", "noise"])
    with pytest.raises(ValueError, match="objective"):
        predictor_class(FEATURES, objective="reg:absoluteerror")
    with pytest.raises(ValueError, match="finish_position"):
        predictor_class(FEATURES).fit(train.drop(columns="finish_position"))


def test_ranker_requires_race_keys_and_valid_finish_positions():
    train, _ = _split(_table())

    with pytest.raises(ValueError, match="season"):
        XGBRankerPredictor(FEATURES).fit(train.drop(columns="season"))
    with pytest.raises(ValueError, match="aantal deelnemers"):
        XGBRankerPredictor(FEATURES).fit(train.assign(finish_position=train["finish_position"] + 1))


def test_xgboost_parameters_override_defaults():
    predictor = XGBRegressorPredictor(FEATURES, max_depth=2)

    assert predictor.params["max_depth"] == 2
    assert predictor.params["learning_rate"] == 0.05
    assert "subsample" not in predictor.params


# --- Interpretatie ---


@pytest.mark.parametrize("predictor_class", XGB_PREDICTORS)
def test_xgboost_feature_importance_highlights_strong_signal(predictor_class):
    train, _ = _split(_table())
    predictor = predictor_class(FEATURES, **FAST).fit(train)

    importance = predictor.feature_importance(train)

    assert importance["feature"].tolist() == FEATURES
    assert importance["gain_share"].sum() == pytest.approx(1.0)
    assert (importance["mean_abs_shap"] >= 0).all()
    signal, noise = importance.set_index("feature").loc[["start_position", "noise"]].to_numpy()
    assert (signal > noise).all()


def test_ridge_feature_importance_returns_standardised_coefficients():
    train, _ = _split(_table())
    predictor = RidgeFeaturePredictor(FEATURES).fit(train)

    importance = predictor.feature_importance(train).set_index("feature")["coefficient"]

    assert importance.index.tolist() == FEATURES
    # Een hogere startpositie (verder achteraan) voorspelt een hogere finish.
    assert importance["start_position"] > abs(importance["noise"]) >= 0


def test_ndcg_ranker_uses_ndcg_objective_with_same_race_groups(monkeypatch):
    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("gestopt na init")

    monkeypatch.setattr(xgb.XGBRanker, "__init__", fake_init)
    train, _ = _split(_table())

    with pytest.raises(RuntimeError, match="gestopt"):
        XGBNDCGRankerPredictor(FEATURES).fit(train)

    assert captured["objective"] == "rank:ndcg"
