from msilib import CAB
from turtle import color


class Vehicle:
    color = 'White'
    def __init__(self, name, max_speed, mileage) -> None:
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage

    def seating_capacity(self, capacity):
        return f"The seating capacity of a {self.name} is {capacity} passengers"

class Bus(Vehicle):
    #Inheritance and Polymorphism (Kế)
    # Use method overriding Sử dụng phương thức ghi đè
    def seating_capacity(self, capacity):
        return super().seating_capacity(capacity)

class Car(Vehicle):
    pass

Car = Car("Mercedes", 180, 12)
print(Car.color, Car.name, "Speed:", Car.max_speed, "Mileafe:", Car.mileage)
