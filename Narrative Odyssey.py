#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[8]:


import numpy as np
import tensorflow as tf

# Define the environment
class StoryGenerationEnvironment:
    def __init__(self, prompt, max_length=100):
        self.prompt = prompt
        self.max_length = max_length
        self.reset()

    def reset(self):
        self.story = self.prompt
        self.step_count = 0

    def get_state(self):
        return self.story

    def step(self, action):
        action_str = str(action)  # Convert action to string
        self.story += " " + action_str
        self.step_count += 1
        done = self.step_count >= self.max_length
        return self.story, 0.0, done

# Define the RL agent
class RLAgent:
    def __init__(self, action_space_size, state_size, vocab_size):
        self.action_space_size = action_space_size
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=64),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(self.action_space_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def select_action(self, state):
        # Tokenize the state and pad if necessary
        state = state.split()[:self.state_size]
        state = [word_to_index.get(word, 0) for word in state]  # Convert words to indices
        state = np.array(state).reshape(1, -1)
        action_probs = self.model.predict(state)[0]
        action = np.random.choice(self.action_space_size, p=action_probs)
        return action

# Training
def train(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.get_state()
        episode_reward = 0

        for t in range(env.max_length):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Update agent (RL training step)

            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        print("Generated Story:")
        print(format_story(env.story))

# Convert numerical story to formatted text
def format_story(story):
    words = []
    for index in story.split():
        if index.isdigit():
            words.append(index_to_word.get(int(index), '<unk>'))  # Convert indices to words, use '<unk>' for unknown indices
        else:
            words.append(index)
    return ' '.join(words)

if __name__ == "__main__":
    # Define the prompt
    prompt = "Once upon a time in a colorful meadow, there was a cheerful bunny named Benny. Benny loved hopping around, exploring the meadow, and making new friends. One sunny morning, Benny met a shy turtle named Timmy. Timmy was looking for his lost shell house, and Benny decided to help him find it."

    # Define a simple vocabulary mapping
    word_to_index = {'Once': 0, 'upon': 1, 'a': 2, 'time': 3, 'in': 4, 'colorful': 5, 'meadow,': 6, 'there': 7, 'was': 8, 'cheerful': 9, 'bunny': 10, 'named': 11, 'Benny.': 12, 'Benny': 13, 'loved': 14, 'hopping': 15, 'around,': 16, 'exploring': 17, 'and': 18, 'making': 19, 'new': 20, 'friends.': 21, 'One': 22, 'sunny': 23, 'morning,': 24, 'met': 25, 'shy': 26, 'turtle': 27, 'Timmy.': 28, 'Timmy': 29, 'looking': 30, 'for': 31, 'his': 32, 'lost': 33, 'shell': 34, 'house,': 35, 'decided': 36, 'to': 37, 'help': 38, 'him': 39, 'find': 40, 'it.': 41, '<unk>': 42}

    # Create index_to_word dictionary
    index_to_word = {index: word for word, index in word_to_index.items()}

    # Initialize the environment and agent
    env = StoryGenerationEnvironment(prompt)
    agent = RLAgent(action_space_size=100, state_size=len(prompt.split()), vocab_size=len(word_to_index))

    # Get the number of episodes from the user
    num_episodes = int(input("Enter the number of episodes: "))

    # Train the agent with the specified number of episodes
    train(env, agent, num_episodes)


# In[ ]:




