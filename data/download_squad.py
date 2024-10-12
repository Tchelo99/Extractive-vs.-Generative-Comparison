import datasets


def download_squad():
    # Download SQuAD dataset
    squad = datasets.load_dataset("squad")

    # Save to disk
    squad.save_to_disk("data/squad")


if __name__ == "__main__":
    download_squad()
    print("SQuAD dataset downloaded successfully.")
