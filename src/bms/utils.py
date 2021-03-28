

def get_file_path(image_id, main_folder='train'):
    return "/workdir/data/bms-molecular-translation/{}/{}/{}/{}/{}.png".format(
        main_folder, image_id[0], image_id[1], image_id[2], image_id
    )
