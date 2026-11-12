"""Pod VLM shake-out: visual association write on a RunPod pod.

Usage:
    python vlm_pod_test.py <proxy_url> [model]

    proxy_url: https://<pod-id>-5000.proxy.runpod.net
    model:     defaults to HuggingFaceTB/SmolVLM-256M-Instruct

Sequence: health -> load model -> vlm_fact_write (red square -> Zorblax)
-> compare on held-out red circle + blue contrast -> eval delta ->
save_model -> download_model into ./vlm_edited_model/
"""

import base64
import io
import sys

from PIL import Image, ImageDraw

from runpod_manager import RunPodManager


def png_b64(color, shape="rect", w=224, h=224):
    im = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(im)
    box = [30, 30, w - 30, h - 30]
    (d.rectangle if shape == "rect" else d.ellipse)(box, fill=color)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    url = sys.argv[1].rstrip("/")
    model = sys.argv[2] if len(sys.argv) > 2 \
        else "HuggingFaceTB/SmolVLM-256M-Instruct"
    rm = RunPodManager()

    print("health:", rm.remote_health(url))
    print("load:", rm.remote_load_model(url, model))

    red = png_b64((200, 0, 0), "rect")
    red2 = png_b64((220, 20, 20), "rect")
    circ = png_b64((200, 0, 0), "circ")   # held-out same color
    blue = png_b64((0, 0, 200), "rect")   # contrast

    print("\n=== base eval ===")
    print(rm.remote_eval(url, "base"))

    print("\n=== vlm_fact_write ===")
    stats = rm.remote_vlm_fact_write(
        url,
        asks=[{"image": red, "q": "What is this symbol?"},
              {"image": red2, "q": "Identify this emblem."}],
        answer="It is the Zorblax sigil, the emblem of the Zorblax order.",
        contrast=[{"image": blue, "q": "What is this symbol?"}],
    )
    print({k: v for k, v in stats.items() if k != "history"})

    print("\n=== held-out compare (red CIRCLE — new shape, same color) ===")
    r = rm.remote_vlm_fact_compare(url, circ, "What is this symbol?")
    if "base" in r:
        print("BASE :", r["base"][:150])
        print("EDIT :", r["edited"][:150])
    else:
        print("RESP:", r)

    print("\n=== contrast compare (blue square) ===")
    r = rm.remote_vlm_fact_compare(url, blue, "What is this symbol?")
    if "edited" in r:
        print("EDIT :", r["edited"][:150])
    else:
        print("RESP:", r)

    print("\n=== post-write eval ===")
    print(rm.remote_eval(url, "post-write"))

    print("\n=== save + download ===")
    print(rm.remote_save_model(url))
    print(rm.remote_download_model(url, "vlm_edited_model"))


if __name__ == "__main__":
    main()
