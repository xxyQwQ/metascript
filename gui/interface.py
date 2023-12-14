import gradio as gr
from .utils import new_path, logging, MetaScript
from PIL import Image
import numpy as np
from time import sleep, time

_TITLE = "Metascript Demo"

_GPU_NUM = 2
_GPU_INDEX = 0

MS = MetaScript('./dataset/script', './checkpoint/generator.pth')

def next_gpu():
    global _GPU_INDEX
    _GPU_INDEX = (_GPU_INDEX + 1) % _GPU_NUM
    return _GPU_INDEX

t2str = lambda t: '{:.3f} ms'.format(t)

def generate(image1, image2, image3, image4, text, tprt, size, width):
    path = new_path()
    gpu = next_gpu() + 1
    info = ''
    if image1 is None or image2 is None or image3 is None or image4 is None:
        info = 'No image uploaded. Randomly chosen.'
        image1, image2, image3, image4 = random_reference()
    if len(text) == 0:
        text = '你没输入，哥。'
    reference = MS.process_reference([Image.fromarray(im) for im in [image1, image2, image3, image4]], path)
    t, tot = time(), 0
    for succ, out in MS.generate(text, reference, size, width, path):
        tot += time() - t
        if not succ:
            yield np.ones((1, 1)), 'Word {} is not supported'.format(out), '-1', t2str(-1)
            return
        yield out, info, gpu, t2str(tot)
        if tprt:
            sleep(0.05)
        t = time()
    
def random_reference():
    image1, image2, image3, image4 = MS.get_random_reference()
    return image1, image2, image3, image4

def launch():
    with gr.Blocks(title=_TITLE) as demo:
        with gr.Row():    
            text2gen = gr.Textbox(label="Text to generate", placeholder="嘿嘿嗨，世界")
        with gr.Row():
            img_ref1 = gr.Image(label="Ref 1")
            img_ref2 = gr.Image(label="Ref 2")
            img_ref3 = gr.Image(label="Ref 3")
            img_ref4 = gr.Image(label="Ref 4")
            with gr.Column():
                tprt = gr.Checkbox(label="Output like typewriter", value=True)
                rand_buttom = gr.Button("Select Random Reference")
                wsize = gr.Slider(
                    label="Word Size",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=64
                    )
                width = gr.Slider(
                    label="Paragraph Width",
                    minimum=200,
                    maximum=3000,
                    step=50,
                    value=1600
                    )
            with gr.Column():
                gpu = gr.Textbox(label="Using GPU", placeholder="-1")
                infert = gr.Textbox(label="Inference time", placeholder="-1")
        with gr.Row():
                gen_buttom = gr.Button("Generate")
        with gr.Row():
            info = gr.Textbox(label="Info", placeholder="")
            
        with gr.Row():
            gen_res = gr.Image(label="Generate Result")
            
        rand_buttom.click(
            fn=random_reference,
            inputs=[],
            outputs=[img_ref1, img_ref2, img_ref3, img_ref4]
        )

        gen_buttom.click(
            fn=generate,
            inputs=[img_ref1, img_ref2, img_ref3, img_ref4, text2gen, tprt, wsize, width],
            outputs=[gen_res, info, gpu, infert]
        )
        
    demo.queue()
    logging.info("Starting server...")
    demo.launch(server_port=8111)
