# Validate input, keep asking user for input until they enter a number between 1 and 10
while True:
    number = int(input("Enter any number b/w 1 and 10: "))
    if 1<=number<=10:
        print("Continue")
    else:
        print("Try again!")
    