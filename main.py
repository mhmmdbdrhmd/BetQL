from utils.train import train
from utils.play import play
import tensorflow as tf 
import warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # For TensorFlow 1.x


def main_menu():
    tf.get_logger().setLevel('ERROR')
    warnings.filterwarnings('ignore')
    agent = None
    env=None
    while True:
        choice = input("Do you want to 'play' or 'train' the RL agent? (play/train/exit): ").strip().lower()
        
        if choice == 'train':
            env,agent = train()
            if agent:
                print("Training completed.")
        elif choice == 'play':
            played = play(env,agent)
            if played:
                print("Returning to main menu.")
        elif choice == 'exit':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 'play', 'train', or 'exit'.")

if __name__ == "__main__":
    main_menu()
