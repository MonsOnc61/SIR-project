class TimepointInfo:
    def __init__(self, 
                 timepoint: str | None = None, 
                 age: float | None = None,
                 edss: float | None = None, 
                 ms_type: str | None = None,
                 lesion_number: int | None = None,
                 lesion_volume: float | None = None):
        self.timepoint = timepoint
        self.age = age
        self.ms_type = ms_type
        self.edss = edss
        self.lesion_number = lesion_number
        self.lesion_volume = lesion_volume
    
    def __str__(self):
        return f"""
timepoint = {self.timepoint},
age = {self.age},
ms_type = {self.ms_type},
edss = {self.edss},
lesion_number = {self.lesion_number},
lesion_volume = {self.lesion_volume}
"""

class PatientInfo:
    def __init__(self, 
                 id: str | None = None, 
                 sex: str | None = None, 
                 timepoint_counter: int = 0,
                 timepoint_infos: dict[str: TimepointInfo] | None = None):
        self.id = id
        self.sex = sex
        self.timepoint_counter = timepoint_counter
        self.timepoint_infos = timepoint_infos
        
    def __str__(self):
        return f"""
id = {self.id},
sex = {self.sex},
timepoint_number = {self.timepoint_counter},
timepoint_infos = {self.timepoint_infos},
"""
