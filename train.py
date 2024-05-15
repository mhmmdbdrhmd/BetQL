from environments.blackjack_env import BlackjackEnv

def main():
    env = BlackjackEnv()
    while True:
        # Start a new game
        observation = env.reset()
        done = False
        
        while not done:
            env.render()
            # Get player's action
            action = input("Enter 'hit' or 'stand': ").strip().lower()
            while action not in ['hit', 'stand']:
                print("Invalid input. Please enter 'hit' or 'stand'.")
                action = input("Enter 'hit' or 'stand': ").strip().lower()

            # Convert action to numeric value
            action = 1 if action == 'hit' else 0
            
            # Step through the environment
            observation, reward, done, info = env.step(action)
            if done:
                # When game ends, show the full dealer hand
                env.render(display_full_dealer=True)
            else:
                env.render()

        # Game has ended
        print("Game over.")
        if reward > 0:
            print("You win!")
        elif reward < 0:
            print("You lose!")
        else:
            print("It's a draw!")
        
        # Ask if the player wants to play again
        play_again = input("Play again? (y/n): ").strip().lower()
        if play_again != 'y':
            break

if __name__ == "__main__":
    main()
