"""
Starter pipelines are built to perform well on the first word in a sentence.
"""

class BaselineStarterPipeline(object):
    def __init__(self):
        self.coldstart = ['The','I','This']

    def get_suggestions(self, text, group):
        if(not text.lstrip()):
            return(self.coldstart)
        text = text.rsplit('.', 1)[-1]
        if(not text.lstrip()):
            return(self.coldstart)
        if(text[-1].isalpha()):
            if(len(text.lstrip().split(' '))==1):
                return(self.coldstart)
        return(['','',''])
