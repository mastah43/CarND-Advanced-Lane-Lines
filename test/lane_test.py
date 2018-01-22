import unittest
import os
from lane import LaneLine


class TestLaneLine(unittest.TestCase):
    def test__is_outlier__no_change(self):
        fit = [1.5094471674962824e-05, -0.013041320689249899, 335.51598894708621]
        fit_last = [1.5094471674962824e-05, -0.013041320689249899, 335.51598894708621]
        self.assertFalse(LaneLine.is_fit_outlier(fit, fit_last))

    def test__is_outlier__tolerable_change(self):
        fit = [1.5094471674962824e-05, -0.013141320689249899, 335.51598894708621]
        fit_last = [1.5094471674962824e-05, -0.013041320689249899, 335.51598894708621]
        self.assertFalse(LaneLine.is_fit_outlier(fit, fit_last))

    def test__is_outlier__triggering_change(self):
        fit = [2.1094471674962824e-05, -0.013041320689249899, 335.51598894708621]
        fit_last = [1.5094471674962824e-05, -0.013141320689249899, 335.51598894708621]
        self.assertTrue(LaneLine.is_fit_outlier(fit, fit_last))

    def test__is_outlier__tolerable_change_but_coeffs_very_different(self):
        fit = [4.0267718316422402e-05, -0.022925891669504327, 330.11584014462466]
        fit_last = [1.5094471674962824e-05, -0.013141320689249899, 335.51598894708621]
        self.assertTrue(LaneLine.is_fit_outlier(fit, fit_last))


if __name__ == '__main__':
    unittest.main()
