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

t2str = lambda t: '{:.3f} s'.format(t)

def generate(image1, image2, image3, image4, text, tprt, tprtspeed, size, width):
    path = new_path()
    gpu = next_gpu() + 1
    info = ''
    ref_imgs = []
    if image1 is None and image2 is None and image3 is None and image4 is None:
        info = 'No image uploaded. Randomly chosen.\n'
        image1, image2, image3, image4 = random_reference()
    ref_imgs = [img for img in [image1, image2, image3, image4] if img is not None]
    while len(ref_imgs) < 4:
        ref_imgs.extend(ref_imgs)
    if len(text) == 0:
        info += 'No text input. Have warned.\n'
        text = '你没输入，哥。'
    if tprt and tprtspeed < 0.5:
        info += 'Typewriter speed too fast. You may fail to see the effect depending on your network delay.\n'
    reference = MS.process_reference([Image.fromarray(im) for im in ref_imgs], path)
    t, tot = time(), 0
    for succ, out in MS.generate(text, reference, size, width, path):
        tot += time() - t
        if not succ:
            yield np.ones((1, 1)), 'Word {} is not supported'.format(out), '-1', t2str(-1)
            return
        yield out, info, gpu, t2str(tot)
        if tprt:
            sleep(tprtspeed)
        t = time()
    
def random_reference():
    image1, image2, image3, image4 = MS.get_random_reference()
    return image1, image2, image3, image4

def launch(port=8111):
    with gr.Blocks(title=_TITLE) as demo:
        with gr.Row():    
            with gr.Column():
                gr.Markdown("# Metascript Demo")
                gr.Markdown("A Chinese Handwriting Generator. Github: https://github.com/xxyQwQ/metascript")
                gr.Markdown("Upload your reference images and input text, then click generate!")
        with gr.Row():    
            text2gen = gr.Textbox(label="Text to generate", placeholder="嘿嘿嗨，世界")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Reference Images")
                gr.Markdown("Upload 1~4 images if you want to use your own style. Otherwise, we will randomly choose some images for you. You can check the randomly chosen images as upload examples.")
                with gr.Row():
                    img_ref1 = gr.Image(label="Ref 1")
                    img_ref2 = gr.Image(label="Ref 2")
                    img_ref3 = gr.Image(label="Ref 3")
                    img_ref4 = gr.Image(label="Ref 4")
                    with gr.Column():
                        gpu = gr.Textbox(label="Using GPU")
                        infert = gr.Textbox(label="Inference time")
                        info = gr.Textbox(label="Info", placeholder="")

        with gr.Row():
            with gr.Column():
                rand_buttom = gr.Button("Select Random Reference")
                with gr.Row():
                    tprt = gr.Checkbox(label="Output like typewriter", value=True)
                    tprtspeed = gr.Slider(
                        label="Typewriter Speed",
                        minimum=0.1,
                        maximum=1,
                        step=0.1,
                        value=0.6
                        )
                wsize = gr.Slider(
                    label="Word Size",
                    minimum=30,
                    maximum=200,
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
            gen_buttom = gr.Button("Generate")
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generate Result")
                gen_res = gr.Image(label="Generate Result", show_label=False)
            
        rand_buttom.click(
            fn=random_reference,
            inputs=[],
            outputs=[img_ref1, img_ref2, img_ref3, img_ref4]
        )

        gen_buttom.click(
            fn=generate,
            inputs=[img_ref1, img_ref2, img_ref3, img_ref4, text2gen, tprt, tprtspeed, wsize, width],
            outputs=[gen_res, info, gpu, infert]
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### About")
                gr.Markdown("This demo is created and maintained by Loping151. See more info at the main site: https://www.loping151.com")
                gr.Markdown("The main server is at Seattle, USA. The mean network delay is about 200ms.")
        
    demo.queue()
    logging.info("Starting server...")
    demo.launch(server_port=port)