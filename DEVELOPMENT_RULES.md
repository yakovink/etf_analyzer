THIS DOCUMENT EXPLAIN HOW WE SHOULD WORK ON THIS PROJECT.
IT AIMED INTO ANY AI AGENT WILL CODE HERE.

The file hierarchy:
There are few folders in the project. any of them have specific and unique role and should not use to any other roles.
For first, there is global_import. global import should contains ALL of the external libraries in the project. other files, except of notebooks, should not have direct import of external library, only from global import.

Next, we have clients folder. clients folder contains classes that should load data from external sources. By external i mean not only web and internet, but also csv/json/db files. except of notebooks, that can do it for explore and test, only clients should have the functionality to get non-memory data from files or web. also, all of ther outputs should be pandas dataframes.

Next, we have processing flder. processing folder have file for each stage that defined in pipelines.md. It should not have check, test, side processing files or any file that not implement onr of those pipelines. if any AI Agent want to test some side processing, it should use a notebook.

Next, we have MLmodules folder. that folder should contains inner stages in pipeline 6, like preproccesing and torch module. it should not contains test or check files. any old or unused file should be deleted in the moment it became unused.

Next, we have data folder. data folder, which wont be backed up on git, contains only data that have to beign writen on the hard drive. if it the main db 'etf_analyzer.db',api key or tables that hard or not ethic to srap from web any time from start.

from last, we have ipynb folder. here agents can open notebooks, explore before they build classes or test builded classes. this is the wild west of the code.

The development methods:
The project is been writen in python language, for now. in continue, we may translate some torch code from python into C++, but not right now.
Python is a very unefficient language. becouse of that, our first role in development with python is to use python a much less as we can. For that, we have multiple libraries that been compiled in C or C++, like pandas, numpy and pytorch.
So, when we want to calc some calculation in data, we will prefere to use build-in verctorial operations. after that, we will prefer to use apply and insert the method we will apply. We will never goes on dataframes or serieses with for loop.

next. we have a db file, like we said before, that contains all of data in sqlite. its only allwed interface is DatabaseManager class. other files can load from it data, not direct from sqlite3. the only files that allowed to open tables or save tables are the pipelines files.

next. all of the code should be type-safe. dont start to validate stubs, but mention for each new parameter/method input/method output its type if its could be Any or Unknown.
that it for now.