def get_path():
    ''' Returns the paths where the data and pretrained network are locaded. This file can be modified as needed.
    '''
    path = {
            "save_dir": "../",
            "scribble_dir": './data/pascal_2012_scribble/',
            "root_voc_dir": './data/VOCdevkit/VOC2012/',
            "feature_dir": './data/Feat/',
            "model_dir": './data/pretrained_model/model/',
        }
    return path