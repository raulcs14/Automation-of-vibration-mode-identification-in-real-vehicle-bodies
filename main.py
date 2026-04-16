"""
Main pipeline: runs static model, modal analysis, and MAC correlation.
"""

from simple_model.analysis.static_model import run_static_model
from simple_model.analysis.modal_analysis import run_modal_analysis
from simple_model.analysis.mac import run_mac_analysis


def main():
    run_static_model()
    run_modal_analysis()
    run_mac_analysis()


if __name__ == "__main__":
    main()
