from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import yaml

def to_gif(images, path):

    images[0].save(path, save_all=True,
                   append_images=images[1:], loop=0, duration=len(images) * 20)


def figure_to_image(figure):

    figure.set_dpi(300)

    figure.canvas.draw()

    return Image.frombytes('RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb())


def image_grid(images, outpath=None, column_titles=None, row_titles=None):

    n_rows = len(images)
    n_cols = len(images[0])

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols, n_rows), squeeze=False)

    for row, _images in enumerate(images):

        for column, image in enumerate(_images):
            ax = axs[row][column]
            ax.imshow(image)
            if column_titles and row == 0:
                ax.set_title(textwrap.fill(
                    column_titles[column], width=12), fontsize='x-small')
            if row_titles and column == 0:
                ax.set_ylabel(row_titles[row], rotation=0, fontsize='x-small', labelpad=1.6 * len(row_titles[row]))
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout(pad=0)
        image = figure_to_image(plt.gcf())
        plt.close()
        return image


def get_module(module, module_name):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 0:
        return module
    else:
        module = getattr(module, module_name[0])
        return get_module(module, module_name[1:])


def set_module(module, module_name, new_module):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)


def freeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = False


def unfreeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = True


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def load_prompts(path):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts