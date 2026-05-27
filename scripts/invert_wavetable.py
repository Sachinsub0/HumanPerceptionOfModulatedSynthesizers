import torch as tr

if __name__ == "__main__":
    path = "../data/listening_test/filter__acid_saw__46_1024.pt"
    wt = tr.load(path, weights_only=True)
    wt_inverted = wt.flip(0)
    save_path = path.replace(".pt", "__inverted.pt")
    tr.save(wt_inverted, save_path)
    print(f"Saved inverted wavetable ({wt_inverted.shape}) to {save_path}")
