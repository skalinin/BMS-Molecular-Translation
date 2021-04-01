import torch


def load_pretrain_model(weights_path, model, device):
    old_model = torch.load(weights_path, device)
    new_dict = model.state_dict()
    old_dict = old_model
    for key, weights in new_dict.items():
        if key in old_dict:
            if new_dict[key].shape == old_dict[key].shape:
                new_dict[key] = old_dict[key]
            else:
                print('\n Weights {} were not loaded'.format(key))
        else:
            print('\n Weights {} were not loaded'.format(key))
    return new_dict


def get_file_path(image_id, main_folder='train'):
    return "/workdir/data/bms-molecular-translation/{}/{}/{}/{}/{}.png".format(
        main_folder, image_id[0], image_id[1], image_id[2], image_id
    )
