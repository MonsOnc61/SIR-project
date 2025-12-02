class PatientInfo:
    def __init__(self, 
                 id: str | None = None, 
                 age: float | None = None,
                 sex: str | None = None, 
                 ms_type: str | None = None,
                 edss: float | None = None, 
                 lesion_number: int | None = None,
                 lesion_volume: float | None = None):
        self.id = id
        self.age = age
        self.sex = sex
        self.ms_type = ms_type
        self.edss = edss
        self.lesion_number = lesion_number
        self.lesion_volume = lesion_volume
        
    def __str__(self):
        return f"""
id = {self.id},
age = {self.age},
sex = {self.sex},
ms_type = {self.ms_type},
edss = {self.edss},
lesion_number = {self.lesion_number},
lesion_volume = {self.lesion_volume}
"""