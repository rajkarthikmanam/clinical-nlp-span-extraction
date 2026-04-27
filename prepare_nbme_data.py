from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parent / "src").resolve()))

from clinical_nlp_span_extraction.nbme import main_prepare_nbme


if __name__ == "__main__":
    main_prepare_nbme()
