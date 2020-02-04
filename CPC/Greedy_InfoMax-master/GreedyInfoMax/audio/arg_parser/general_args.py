
def parse_general_args(parser):
    parser.add_option(
        "--num_epochs", type="int", default=1000, help="Number of Epochs for Training"
    )
    parser.add_option("--seed", type="int", default=2, help="Random seed for training")
    parser.add_option(
        "--batch_size",
        type="int",
        default=8,
        help="Batchsize",
    )
    parser.add_option(
        "-i",
        "--data_input_dir",
        type="string",
        default="/vol/medic01/users/av2514/Pycharm_projects/crystal_cone/data/",
        help="Directory to store bigger datafiles (dataset and audio_models)",
    )
    parser.add_option(
        "-o",
        "--data_output_dir",
        type="string",
        default=".",
        help="Directory to store bigger datafiles (dataset and audio_models)",
    )
    parser.add_option(
        "--validate",
        action="store_true",
        default=False,
        help="Boolean to decide whether to split train dataset into train/val and plot validation loss",
    )
    parser.add_option(
        "--SVM_training_samples",
        type="int",
        default=0,
        help="If value is > 0, train an SVM on the representation of the model after every epoch of training "
             "and test speaker classification performance on testset (20 is a reasonable value here)",
    )
    parser.add_option(
        "--save_dir",
        type="string",
        default="",
        help="If given, uses this string to create directory to save results in "
             "(be careful, this can overwrite previous results); "
             "otherwise saves logs according to time-stamp",
    )

    return parser
