
#oop

class XRayImage:
    def __init__(self, patient_name, body_part):
        self.patient_name = "Ali"     # store patient name
        self.body_part = "Heart"        # store body part

    def describe(self):
        print(f"X-Ray of {self.body_part} for patient {self.patient_name}")

# Create an object
xray1 = XRayImage(patient_name="Ali", body_part="Chest")
xray1.describe()