# Coffee Customization 
# Problem: Customize a coffee order: "Small", "Medium" or "Large" with an option for "Extra Shot"
# Esspresso
coffee_size = input("Enter type of coffee: 'Small/Medium/Large'\n")
extras = input("Do you want an Extra Shot of espresso? ('y' for Yes. 'N' for No)\n ")

if extras=='y':
    if coffee_size == 'Small':
        print(f'Woohoo! {coffee_size} coffee with extra espresso ordered!')
    elif coffee_size == 'Medium':
        print(f'Woohoo! {coffee_size} coffee with extra espresso ordered!')
    elif coffee_size=='Large':
        print(f'Woohoo! {coffee_size} coffee with extra espresso ordered!')
    else:
        print("Select valid choice. Small/Medium/Large.")
else:
     if coffee_size == 'Small':
        print(f'Woohoo! {coffee_size} coffee ordered!')
     elif coffee_size == 'Medium':
        print(f'Woohoo! {coffee_size} coffee ordered!')
     elif coffee_size=='Large':
        print(f'Woohoo! {coffee_size} coffee ordered!')
     else:
        print("Select valid choice. Small/Medium/Large.")