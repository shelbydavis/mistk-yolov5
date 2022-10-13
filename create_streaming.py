import base64
import os
import io
import sys
import json
from PIL import Image

if __name__ == "__main__":
    filename = sys.argv[1]
    byteImgIO = io.BytesIO()
    byteImg = Image.open(filename)
    byteImg.save(byteImgIO, "PNG")
    byteImgIO.seek(0)
    out_dict = dict()
    out_dict[filename] = (base64.b64encode(byteImgIO.read())).decode(encoding="utf-8")
    output = sys.argv[2]
    with open(output, "w") as outfile :
        outfile.write(json.dumps(out_dict))
    