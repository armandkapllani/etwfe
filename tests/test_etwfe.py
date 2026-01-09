"""Tests for ETWFE package."""

import numpy as np
import pandas as pd
import pytest

from etwfe import ETWFE, etwfe


@pytest.fixture
def sample_panel_data():
    """Create sample panel data for testing."""
    np.random.seed(42)

    n_units = 100
    n_periods = 10

    # Create balanced panel
    data = []
    for i in range(n_units):
        # Assign treatment cohort (some never treated)
        if i < 30:
            first_treat = 5  # Early adopters
        elif i < 60:
            first_treat = 7  # Late adopters
        else:
            first_treat = 99  # Never treated (control)

        for t in range(n_periods):
            # Treatment indicator
            treated = 1 if t >= first_treat else 0

            # Outcome with unit FE, time FE, and treatment effect
            unit_fe = np.random.normal(0, 1)
            time_fe = t * 0.1
            treatment_effect = 2.0 * treated
            noise = np.random.normal(0, 0.5)

            y = unit_fe + time_fe + treatment_effect + noise

            data.append(
                {
                    "id": i,
                    "year": 2000 + t,
                    "first_treat": 2000 + first_treat,
                    "y": y,
                    "x1": np.random.normal(0, 1),
                }
            )

    return pd.DataFrame(data)


class TestETWFE:
    """Test ETWFE class."""

    def test_init_basic(self, sample_panel_data):
        """Test basic initialization."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
        )
        assert model.fml == "y ~ 0"
        assert model.tvar == "year"
        assert model.gvar == "first_treat"

    def test_init_with_controls(self, sample_panel_data):
        """Test initialization with control variables."""
        model = ETWFE(
            fml="y ~ x1",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
        )
        assert model._ctrls == ["x1"]

    def test_init_missing_column(self, sample_panel_data):
        """Test error on missing column."""
        with pytest.raises(ValueError, match="Missing columns"):
            ETWFE(
                fml="y ~ nonexistent",
                tvar="year",
                gvar="first_treat",
                data=sample_panel_data,
            )

    def test_fit(self, sample_panel_data):
        """Test model fitting."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            ivar="id",
        )
        model.fit()

        assert model.model_ is not None
        assert model._fit_data is not None
        assert model.formula_ is not None

    def test_emfx_simple(self, sample_panel_data):
        """Test simple ATT computation."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            ivar="id",
        )
        model.fit()

        att = model.emfx(type="simple")

        assert len(att) == 1
        assert "estimate" in att.columns
        assert "std.error" in att.columns
        assert "conf.low" in att.columns
        assert "conf.high" in att.columns

    def test_emfx_event(self, sample_panel_data):
        """Test event study computation."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            ivar="id",
        )
        model.fit()

        event = model.emfx(type="event")

        assert len(event) > 0
        assert "event" in event.columns
        assert "estimate" in event.columns

    def test_emfx_group(self, sample_panel_data):
        """Test cohort-specific effects."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            ivar="id",
        )
        model.fit()

        group = model.emfx(type="group")

        assert len(group) > 0
        assert "first_treat" in group.columns
        assert "estimate" in group.columns

    def test_emfx_calendar(self, sample_panel_data):
        """Test calendar time effects."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            ivar="id",
        )
        model.fit()

        calendar = model.emfx(type="calendar")

        assert len(calendar) > 0
        assert "year" in calendar.columns
        assert "estimate" in calendar.columns


class TestETWFEFunction:
    """Test etwfe convenience function."""

    def test_etwfe_function(self, sample_panel_data):
        """Test convenience function."""
        model = etwfe(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            ivar="id",
        )

        assert model.model_ is not None
        assert isinstance(model, ETWFE)

    def test_etwfe_with_controls(self, sample_panel_data):
        """Test with control variables."""
        model = etwfe(
            fml="y ~ x1",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            ivar="id",
        )

        att = model.emfx(type="simple")
        assert len(att) == 1


class TestControlGroups:
    """Test different control group specifications."""

    def test_notyet_control(self, sample_panel_data):
        """Test not-yet-treated control group."""
        model = etwfe(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            cgroup="notyet",
        )

        assert model.cgroup == "notyet"
        att = model.emfx(type="simple")
        assert len(att) == 1


class TestReferences:
    """Test reference level handling."""

    def test_auto_references(self, sample_panel_data):
        """Test automatic reference selection."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
        )

        assert model.tref is not None
        assert model.gref is not None

    def test_manual_references(self, sample_panel_data):
        """Test manual reference specification."""
        model = ETWFE(
            fml="y ~ 0",
            tvar="year",
            gvar="first_treat",
            data=sample_panel_data,
            tref=2000,
            gref=2099,
        )

        assert model.tref == 2000
        assert model.gref == 2099


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
