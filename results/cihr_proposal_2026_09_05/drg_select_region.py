"""Select an exclusion polygon interactively or reproduce a saved DRG selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.draw import polygon
from drg_inference import PANEL, check_outputs, sha


def exclusion_mask(vertices, shape):
    xy = np.asarray(vertices)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise ValueError('Need at least three polygon vertices')
    excluded = np.zeros(shape, bool)
    rr, cc = polygon(xy[:, 1], xy[:, 0], shape)
    excluded[rr, cc] = True
    if not excluded.any() or excluded.all():
        raise ValueError('Select a nonempty region smaller than the image')
    return excluded


def save_region(base, out, vertices, replay=None):
    image_path, mask_path = base/'images/drg_composite.png', base/'masks/drg_composite.png'
    image, mask = np.array(Image.open(image_path)), np.array(Image.open(mask_path))
    assert image.shape[:2] == mask.shape[:2]
    hashes = {str(p): sha(p) for p in [image_path, mask_path]}
    if replay:
        assert list(image.shape[:2]) == replay['shape']
        assert sorted(hashes.values()) == sorted(replay['original_sha256'].values())
    excluded = exclusion_mask(vertices, image.shape[:2])
    for sub, array in [('images', image), ('masks', mask)]:
        target = out/sub/'drg_composite.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        result = array.copy()
        result[excluded] = 0
        Image.fromarray(result).save(target)
        assert np.array_equal(np.array(Image.open(target)), result)
    Image.fromarray(excluded.astype(np.uint8)*255).save(out/'exclusion.png')
    metadata = dict(vertices_xy=vertices, shape=list(excluded.shape), excluded_pixels=int(excluded.sum()),
                    original_sha256=hashes, masked_image_sha256=sha(out/'images/drg_composite.png'),
                    masked_annotation_sha256=sha(out/'masks/drg_composite.png'))
    if replay:
        for key in ['excluded_pixels', 'masked_image_sha256', 'masked_annotation_sha256']:
            assert metadata[key] == replay[key], key
    (out/'region.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PANEL)
    parser.add_argument('--output-dir', type=Path, help='Defaults to input-dir/masked_region.')
    parser.add_argument('--region', type=Path, help='Replay this region.json without opening the GUI.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    base = args.input_dir.resolve()
    out = (args.output_dir or base/'masked_region').resolve()
    if out == base:
        raise ValueError('Output directory must differ from the source input directory')
    check_outputs([out/'images', out/'masks', out/'exclusion.png', out/'region.json'], args.overwrite)
    if args.region:
        replay = json.loads(args.region.read_text())
        save_region(base, out, replay['vertices_xy'], replay)
        print('Replayed region:', out)
        return
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector
    image = np.array(Image.open(base/'images/drg_composite.png'))
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title('Click polygon vertices; close polygon. Enter: save. Escape: reset.')
    preview = ax.imshow(np.zeros((*image.shape[:2], 4)))
    state = dict(vertices=None)

    def selected(vertices):
        state['vertices'] = vertices
        rgba = np.zeros((*image.shape[:2], 4))
        rgba[exclusion_mask(vertices, image.shape[:2])] = [1, 1, 0, .35]
        preview.set_data(rgba)
        fig.canvas.draw_idle()

    selector = PolygonSelector(ax, selected, props=dict(color='yellow', linewidth=1.5))

    def key(event):
        if event.key == 'escape':
            selector.clear()
            state['vertices'] = None
            preview.set_data(np.zeros((*image.shape[:2], 4)))
            fig.canvas.draw_idle()
        elif event.key == 'enter' and state['vertices'] is not None:
            save_region(base, out, state['vertices'])
            print('Saved region:', out)
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', key)
    plt.show()


if __name__ == '__main__':
    main()
