import torch


def join_inchi_layers(prev_joined_inchis, curr_inchi_layers, chem_token):
    tokened_curr_inchi_layers = []
    for curr_inchi_layer in curr_inchi_layers:
        if len(curr_inchi_layer) > 0:
            tokened_curr_inchi_layers.append(chem_token + curr_inchi_layer)

    joined_inchis = [i + j for i, j in zip(prev_joined_inchis,
                                           tokened_curr_inchi_layers)]
    return joined_inchis


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
