# Fruit Ripeness calculator- Determine if a fruit is ripe, overripe, or unique based on its color 
# (e.g. Banana, Green - Unripe, Yellow-Ripe, Brown-overripe.)

fruit = "Banana"
fruit_color = str(input("Enter color of fruit: Green/Yellow/Brown. "))
if fruit_color=='Green':
    ripeness = 'Unripe'
elif fruit_color == 'Yellow':
    ripeness = 'Ripe'
elif fruit_color=='Brown':
    ripeness = 'Overripe'
else:
    ripeness = "banana, observe your fruit with open eyes!"
print(f"Your {fruit} is {ripeness}")