from ultralytics import SAM
model = SAM("sam2.1_b.pt")
print(model.predict.__doc__)
