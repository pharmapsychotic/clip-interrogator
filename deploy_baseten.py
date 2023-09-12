import os

import baseten
import truss

model = truss.load("./truss")
baseten.login(os.environ["BASETEN_API_KEY"])
baseten.deploy(model, model_name="CLIP Interrogator", publish=True)
