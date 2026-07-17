import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, default_sphere
from dipy.reconst.dti import TensorModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.direction import peaks_from_model
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking import utils
from dipy.viz import actor, window

try:
    from fury.colormap import line_colors
except ImportError:
    try:
        from dipy.viz.colormap import line_colors
    except ImportError:
        def line_colors(streamlines, cmap='rgb_standard'):
            from dipy.viz import colormap as cmap_module
            return cmap_module.create_colormap(
                np.array([len(s) for s in streamlines], dtype=float)
            )

print("============================================================================================\n")
namaprogram= "DIPY Neurosains Brain Analysis\n"
devby="Developed by Ananda Rauf Maududi\n"
devdate="Developed date 17 July 2026\n"
print("============================================================================================\n")

def download_and_load_data():
    fetch_stanford_hardi()
    img, gtab = read_stanford_hardi()
    return img, gtab

def pipeline_tractography():
    img, gtab = download_and_load_data()
    data = img.get_fdata()
    affine = img.affine
    
    background_mask = data[..., 0] > 50
    
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(data, mask=background_mask)
    fa = tensor_fit.fa
    
    stopping_criterion = ThresholdStoppingCriterion(fa, 0.2)
    
    direction_peaks = peaks_from_model(
        model=tensor_model,
        data=data,
        sphere=default_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=background_mask,
        return_odf=False,
        normalize_peaks=True
    )
    
    seed_mask = np.zeros(fa.shape, dtype=bool)
    seed_mask[40:60, 40:60, 35:45] = True
    seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)
    
    streamlines_generator = LocalTracking(
        direction_peaks,
        stopping_criterion,
        seeds,
        affine,
        step_size=0.5
    )
    
    return list(streamlines_generator)

def render_native_window(streamlines):
    scene = window.Scene()
    
    colors = line_colors(streamlines)
    streamline_actor = actor.line(streamlines, colors=colors)
    scene.add(streamline_actor)
    
    window.show(scene, size=(1200, 900), title="3D Brain Tractography - Ananda Technology Solution")

if __name__ == "__main__":
    brain_streamlines = pipeline_tractography()
    render_native_window(brain_streamlines)