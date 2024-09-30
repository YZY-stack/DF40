import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR
from .utils import slowfast

from .xception_detector import XceptionDetector
from .spsl_detector import SpslDetector
from .srm_detector import SRMDetector
from .recce_detector import RecceDetector
from .clip_detector import CLIPDetector
from .sbi_detector import SBIDetector
from .i3d_detector import I3DDetector
from .rfm_detector import RFMDetector
