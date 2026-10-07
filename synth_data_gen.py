"""
Synthetic bouncing-balls dataset generator.

Produces a numpy array of shape (num_sequences, seq_len, img_size, img_size)
of grayscale frames showing one or more balls bouncing inside a box.

Usage:
    python generate_bouncing_balls.py
    # -> writes bouncing_balls.npz (train + val splits) in the current dir

Then in your training code:
    data = np.load("bouncing_balls.npz")
    train, val = data["train"], data["val"]   # (N, T, H, W) uint8
"""

import numpy as np
import cv2
import imageio


def generate_bouncing_balls_dataset(
    num_sequences: int = 500,
    seq_len: int = 20,
    img_size: int = 64,
    num_balls: int = 1,
    radius_range: tuple[int, int] = (4, 8),
    speed_range: tuple[float, float] = (2.0, 5.0),
    seed: int = 0,
) -> np.ndarray:
    """Generate a dataset of bouncing-ball sequences.

    Each ball moves in a straight line and reflects off the walls of the
    frame. No inter-ball collisions -- keeps the dynamics simple and the
    supervision signal clean, which is what you want for a first world model.
    """
    rng = np.random.default_rng(seed)
    dataset = np.zeros((num_sequences, seq_len, img_size, img_size), dtype=np.uint8)

    for seq_idx in range(num_sequences):
        balls = []
        for _ in range(num_balls):
            r = float(rng.integers(*radius_range))
            pos = rng.uniform(r, img_size - r, size=2)
            angle = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(*speed_range)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            balls.append({"pos": pos, "vel": vel, "r": r})

        for t in range(seq_len):
            frame = np.zeros((img_size, img_size), dtype=np.uint8)
            for ball in balls:
                ball["pos"] = ball["pos"] + ball["vel"]

                # reflect off walls on each axis independently
                for dim in range(2):
                    if ball["pos"][dim] - ball["r"] < 0:
                        ball["pos"][dim] = ball["r"]
                        ball["vel"][dim] *= -1
                    elif ball["pos"][dim] + ball["r"] > img_size:
                        ball["pos"][dim] = img_size - ball["r"]
                        ball["vel"][dim] *= -1

                center = tuple(ball["pos"].astype(int))
                cv2.circle(frame, center, int(ball["r"]), color=255, thickness=-1)

            dataset[seq_idx, t] = frame

    return dataset


def save_as_gif(sequence: np.ndarray, path: str, fps: int = 10) -> None:
    """Quick visual sanity check: dump one sequence as an animated GIF.

    Requires `imageio` (pip install imageio). Not needed for training,
    just for eyeballing that the physics looks right.
    """
    

    frames = [sequence[t] for t in range(sequence.shape[0])]
    imageio.mimsave(path, frames, fps=fps)


if __name__ == "__main__":
    # Tweak num_balls / img_size / seq_len here as you iterate.
    train = generate_bouncing_balls_dataset(num_sequences=800, seed=0)
    val = generate_bouncing_balls_dataset(num_sequences=100, seed=1)

    np.savez_compressed("bouncing_balls.npz", train=train, val=val)
    print(f"train: {train.shape}, val: {val.shape}")

    # Optional: sanity-check GIF of the first training sequence
    try:
        save_as_gif(train[0], "sample_sequence.gif")
        print("Wrote sample_sequence.gif for a visual check")
    except ImportError:
        print("Install imageio to also get a sanity-check GIF: pip install imageio")