import ultralytics
ultralytics.data.converter.convert_coco(
    labels_dir='/home/aditya/snaglist_dataset_sep4_high_quality_robo_train_insta_val_multicls',
    # labels_dir = '/home/aditya/snaglist_semantic_marking_1-158/annotations',
    save_dir='/home/aditya/snaglist_dataset_sep4_high_quality_robo_train_insta_val_multicls',
    # save_dir = '/home/aditya/snaglist_semantic_marking_1-158/valid',
    use_segments=False,
    use_keypoints=False, cls91to80=False)
