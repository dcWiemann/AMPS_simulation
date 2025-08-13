#!/usr/bin/env python3
"""
Simple script to run the workflow test to see full output.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from tests.test_full_workflow import TestFullWorkflow
from amps_simulation.core.components import Component

if __name__ == "__main__":
    # Clear component registry manually
    Component.clear_registry()
    
    test_instance = TestFullWorkflow()
    test_instance.test_full_workflow_all_circuits()