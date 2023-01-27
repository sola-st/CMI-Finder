import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    hlep="specifiy the name of the model you want to test, one of three options: bilstm, triplet, codet5",
    required=True
)

parser.add_argument(
    "--source",
    help="specify the path of the model",
    required=True
)

parser.add_argument(
    "--path_data",
    help="path to the test file or folder or extracted pairs"
)

parser.add_argument(
    "--path_labels",
    help="path to labels"
)

if __name__ == "__main__":
    args = parser.parse_args()

    model = args.model
    source = args.source
    test_type = args.test_type
    path = args.path


    if model == "bilstm":
        pass
    elif model == "triplet":
        pass
    elif model == "codet5":
        pass
    else:
        raise ValueError("Expected one of three options: bilstm, triplet, codet5")

