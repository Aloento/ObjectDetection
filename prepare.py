from VOCDataset import VOCDataset


def prepare():
    var = VOCDataset()

    for i in var:
        print(i)

    print(var)


if __name__ == "__main__":
    prepare()
