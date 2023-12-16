import gradio as gr
from .utils import new_path, logging, convert_input_text, MetaScript
from PIL import Image
import numpy as np
from time import sleep, time

_TITLE = "Metascript Demo"

_GPU_NUM = 2
_GPU_INDEX = 0

css = """
.centered-container {
    display: flex;
    justify-content: center;
    align-items: center;
}
"""

MS = MetaScript('./dataset/script', './checkpoint/generator.pth')

def next_gpu():
    global _GPU_INDEX
    _GPU_INDEX = (_GPU_INDEX + 1) % _GPU_NUM
    return _GPU_INDEX

t2str = lambda t: '{:.3f} s'.format(t)

def clear():
    return None, None, None, None

def generate(image1, image2, image3, image4, text, tprt, tprtspeed, size, width):
    logging.info('Get a generate request.')
    if tprt and len(text) > 100//tprtspeed or len(text) > 1000:
        tprt = False
        text = text[:1000]
        yield np.ones((1, 1)), 'Too long! you are occupying to mush resource! Current length is: {}/{} or {} with typewriter on.'.format(len(text), 1000, 100//tprtspeed), '-1', t2str(-1)

    info = ''
    path = new_path()
    gpu = next_gpu() + 1

    logging.info('Using path: {}'.format(path))
    logging.info('Input text: {}'.format(text))
    text, conv_list = convert_input_text(text)
    if len(conv_list) > 0:
        info += 'Converted not supported character(s): {}\n'.format(conv_list)
    logging.info('Converted text: {}'.format(text))


    ref_imgs = []
    if image1 is None and image2 is None and image3 is None and image4 is None:
        info += 'No image uploaded. Randomly chosen.\n'
        logging.info('No image uploaded. Randomly chosen.')
        image1, image2, image3, image4 = random_reference()
    ref_imgs = [img for img in [image1, image2, image3, image4] if img is not None]
    while len(ref_imgs) < 4:
        ref_imgs.extend(ref_imgs)
    if len(text) == 0:
        info += 'No text input. Have warned.\n'
        text = '你没输入，哥。'
    if tprt and tprtspeed < 0.5:
        info += 'Typewriter speed too fast. You may fail to see the effect depending on your network delay.\n'

    if info.endswith('\n'):
        info = info[:-1]

    reference = MS.process_reference([Image.fromarray(im) for im in ref_imgs], path)
    t, tot = time(), 0
    for succ, out in MS.generate(text, reference, size, width, path):
        tot += time() - t
        if not succ:
            yield None, 'Word {} is not supported'.format(out), '-1', t2str(-1)
            return
        yield out, info, gpu, t2str(tot)
        if tprt:
            sleep(tprtspeed)
        t = time()
    
def random_reference():
    logging.info('Get a random reference request.')
    image1, image2, image3, image4 = MS.get_random_reference()
    return image1, image2, image3, image4

def launch(port=8111):
    with gr.Blocks(title=_TITLE, css=css) as demo:
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
                    with gr.Column():
                        with gr.Row():
                            img_ref1 = gr.Image(label="Ref 1")
                            img_ref2 = gr.Image(label="Ref 2")
                        with gr.Row():
                            img_ref3 = gr.Image(label="Ref 3")
                            img_ref4 = gr.Image(label="Ref 4")
                    with gr.Column():
                        with gr.Row():
                            gpu = gr.Textbox(label="Using GPU")
                            infert = gr.Textbox(label="Inference time")
                        info = gr.Textbox(label="Info", placeholder="")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    rand_buttom = gr.Button("Select Random Reference")
                    clear_bottom = gr.Button("Clear Current Reference")
                with gr.Row():
                    tprt = gr.Checkbox(label="Output like typewriter", value=True)
                    tprtspeed = gr.Slider(
                        label="Typewriter Speed",
                        minimum=0,
                        maximum=2,
                        step=0.05,
                        value=1
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

        clear_bottom.click(
            fn=clear,
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
                gr.Markdown("GUI version: 1.1")
                gr.Markdown("This demo is created and maintained by Loping151. See more info at the main site: https://www.loping151.com")
                gr.Markdown("The main server is at Seattle, USA. The mean network delay is about 200ms.")
                gr.Markdown("#### You may fail to see the typewriter effect when the network delay is too long or the server is too busy. If this happens, you should refresh the page and turn off the typewriter effect for faster generation.")
        
    demo.queue()
    logging.info("Starting server...")
    demo.launch(server_port=port)
