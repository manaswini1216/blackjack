import streamlit as st
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

# App title
st.set_page_config(layout="centered")
st.title("🃏 Blackjack Q-Learning Agent (Trained)")

# Load the trained Q-table
@st.cache_data
def load_q_table():
    with open("q_table.pkl", "rb") as f:
        Q = pickle.load(f)
    return Q

Q = load_q_table()

# Create Blackjack environment
env = gym.make("Blackjack-v1", render_mode="rgb_array")

# Button to run an episode
if st.button("▶️ Play 1 Episode"):
    state, _ = env.reset()
    total_reward = 0
    frames = []
    done = False

    while not done:
        action = np.argmax([Q.get((state, a), 0.0) for a in [0, 1]])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        frames.append(env.render())
        state = next_state

    # Show result
    st.subheader(f"🏁 Episode finished! Reward: **{total_reward}**")
    if total_reward > 0:
        st.success("You won! 🎉")
    elif total_reward < 0:
        st.error("You lost 😢")
    else:
        st.info("It's a draw.")

    # Show frames
    st.subheader("🎞️ Gameplay Frames")
    for frame in frames:
        fig, ax = plt.subplots()
        ax.imshow(frame)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

env.close()
