import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.explainability.templates as templates

print("Attributes of templates:", dir(templates))
if hasattr(templates, "get_personalized_insights"):
    print("Function get_personalized_insights is present!")
else:
    print("Function get_personalized_insights is missing!")
    with open(templates.__file__, "r", encoding="utf-8") as f:
        print("File contents:\n" + f.read())
