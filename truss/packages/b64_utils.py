import base64
from io import BytesIO

from PIL import Image

get_preamble = lambda fmt: f"data:image/{fmt.lower()};base64,"


def pil_to_b64(pil_img, format="PNG"):
    buffered = BytesIO()
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue())
    return get_preamble(format) + str(img_str)[2:-1]


def b64_to_pil(b64_str, format="PNG"):
    return Image.open(
        BytesIO(base64.b64decode(b64_str.replace(get_preamble(format), "")))
    )
