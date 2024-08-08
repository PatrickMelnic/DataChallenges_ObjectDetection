import sys, os
sys.path.append("big_vision")

from scenic.projects.owl_vit import main

main(workdir="temp", config=os.path.join("clip_pretraining", "trained_clip_for_owlvit.py"))


