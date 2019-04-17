from os.path import expanduser

class Dataset:

    def __init__(self):
        self.dataset_path = None
        self.ground_truth = None
        self.camera_file = None
        self.camera = None
        self.quat = None
        self.init_pose = None
        self.rgb_image = None
        self.pre_assoc_file_path = None
        self.ate_associate_identity = None
        self.descr = None
        self.volume_size = None

    # Correctly interpret ~ as the home directory in paths.
    def fix_paths(self):
        if self.dataset_path:
            self.dataset_path = expanduser(self.dataset_path)
        if self.ground_truth:
            self.ground_truth = expanduser(self.ground_truth)
        if self.camera_file:
            self.camera_file = expanduser(self.camera_file)
        if self.rgb_image:
            self.rgb_image = expanduser(self.rgb_image)
        if self.pre_assoc_file_path:
            self.pre_assoc_file_path = expanduser(self.pre_assoc_file_path)
        
