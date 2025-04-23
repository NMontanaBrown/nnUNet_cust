import os
import subprocess
import sys
from setuptools import setup, find_namespace_packages, Command

class InstallCustomTorch(Command):
    """Custom command to install PyTorch with MPS support from GitHub"""
    description = 'Install custom PyTorch with MPS support'
    user_options = []
    
    def initialize_options(self):
        pass
        
    def finalize_options(self):
        pass
        
    def run(self):
        # Try to uninstall existing torch if present
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch'])
            print("Uninstalled existing PyTorch")
        except:
            print("No existing PyTorch installation found")
            
        # Create a temporary directory for cloning PyTorch
        temp_dir = os.path.join(os.getcwd(), 'pytorch_temp')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            # Clone the repository
            subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', 'convtranspose_mps_remove_check', 
                                'https://github.com/NMontanaBrown/pytorch.git', temp_dir])
            
            # Change to the repository directory
        except Exception as e:
            print(f"Failed to clone PyTorch: {e}")
            raise
        try:
            cwd = os.getcwd()
            os.chdir(temp_dir)
            
            # Update submodules
            subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])

            # Install requirements and PyTorch
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            subprocess.check_call([sys.executable, 'setup.py', 'develop'])
            
            # Change back to the original directory
            os.chdir(cwd)
            print("Successfully installed custom PyTorch with MPS support!")
            
        except Exception as e:
            print(f"Failed to install custom PyTorch: {e}")
            raise
        finally:
            # Clean up if needed
            # Comment out if you want to keep the repository
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            pass

setup(name='nnunet',
      packages=find_namespace_packages(include=["nnunet", "nnunet.*"]),
      version='1.7.0',
      description='nnU-Net. Framework for out-of-the box biomedical image segmentation.',
      url='https://github.com/MIC-DKFZ/nnUNet',
      author='Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators==0.21",
            "numpy",
            "scikit-learn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 
            "tifffile", 
            "matplotlib",
      ],
      entry_points={
          'console_scripts': [
              'nnUNet_convert_decathlon_task = nnunet.experiment_planning.nnUNet_convert_decathlon_task:main',
              'nnUNet_plan_and_preprocess = nnunet.experiment_planning.nnUNet_plan_and_preprocess:main',
              'nnUNet_train = nnunet.run.run_training:main',
              'nnUNet_train_DP = nnunet.run.run_training_DP:main',
              'nnUNet_train_DDP = nnunet.run.run_training_DDP:main',
              'nnUNet_predict = nnunet.inference.predict_simple:main',
              'nnUNet_ensemble = nnunet.inference.ensemble_predictions:main',
              'nnUNet_find_best_configuration = nnunet.evaluation.model_selection.figure_out_what_to_submit:main',
              'nnUNet_print_available_pretrained_models = nnunet.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'nnUNet_print_pretrained_model_info = nnunet.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'nnUNet_download_pretrained_model = nnunet.inference.pretrained_models.download_pretrained_model:download_by_name',
              'nnUNet_download_pretrained_model_by_url = nnunet.inference.pretrained_models.download_pretrained_model:download_by_url',
              'nnUNet_determine_postprocessing = nnunet.postprocessing.consolidate_postprocessing_simple:main',
              'nnUNet_export_model_to_zip = nnunet.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'nnUNet_install_pretrained_model_from_zip = nnunet.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'nnUNet_change_trainer_class = nnunet.inference.change_trainer:main',
              'nnUNet_evaluate_folder = nnunet.evaluation.evaluator:nnunet_evaluate_folder',
              'nnUNet_plot_task_pngs = nnunet.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet'],
      cmdclass={
          'install_torch': InstallCustomTorch,
      }
      )
