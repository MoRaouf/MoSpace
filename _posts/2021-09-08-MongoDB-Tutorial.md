---
toc: true
layout: post
description: "Document-oriented NoSQL Database Turorial & Code Snippets"
categories: [post]
tags: [MongoDB, NoSQL]
title: "Introduction to MongoDB"
image: "images/posts/MongoDB.png"
comments: true
featured: false

---

- Collection - an organized store of documents in MongoDB usually with common fields between documents. There can be many collections per database and many documents per collection. (Equal to Table in RDBMS)
- *Document* - a way to organize and store data as a set of field-value pairs (in JSON format).
    - *Field* - a unique identifier for a datapoint.
    - *Value* - data related to a given identifier.
- *Replica Set* - a few connected machines that store the same data to ensure that if something happens to one of the machines the data will remain intact. Comes from the word replicate - to copy something.
- *Instance* - a single machine locally or in the cloud, running a certain software, in our case it is the MongoDB database.
- *Cluster* - group of servers that store your data.
- Namespace - The concatenation of the database name and collection name is called a namespace. (e.g., samples.cars)

- MongoDB stores data in BSON format both internally, and over the network, but that doesn’t mean you can’t think of MongoDB as a JSON database. Anything you can represent in JSON can be natively stored in MongoDB, and retrieved just as easily in JSON.

![MongoDB%20Tutorial%207ad9850475cf4b79b84356f38a6c5ea2/RDBMS_vs_MongoDB.png](MongoDB%20Tutorial%207ad9850475cf4b79b84356f38a6c5ea2/RDBMS_vs_MongoDB.png)

**— How to connect to MongoDB shell "For Atlas":**

Go to Atlas, connect to Sandbox (or any created cluster) & choose connect through shell. Then copy the command & paste it in Command Prompt & enter password. Then work through the command prompt & type any MongoDB command.

- General format in M001 course: `mongo "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<collection>"`
My Command : 
`mongo "mongodb+srv://m001-student:m001-mongodb-basics@sandbox.q8r0r.mongodb.net"`
- Connect to MongoDB Compass : `mongodb+srv://m001-student:m001-mongodb-basics@sandbox.q8r0r.mongodb.net`

- **Importing & Exporting:**  Better to import using compass as its very easy.

![MongoDB%20Tutorial%207ad9850475cf4b79b84356f38a6c5ea2/MongoDB_Atlas_Import.png](MongoDB%20Tutorial%207ad9850475cf4b79b84356f38a6c5ea2/MongoDB_Atlas_Import.png)

![MongoDB%20Tutorial%207ad9850475cf4b79b84356f38a6c5ea2/MongoDB_Atlas_Import_2.png](MongoDB%20Tutorial%207ad9850475cf4b79b84356f38a6c5ea2/MongoDB_Atlas_Import_2.png)

- mongoimport : Import JSON file
    - `mongoimport --uri mongodb+srv://m001-student:<PASSWORD>@sandbox.q8r0r.mongodb.net/<DATABASE> --collection <COLLECTION> --type <FILETYPE> --file <FILENAME>`
    - Import a local file to Atlas : 
    `mongoimport --uri mongodb+srv://m001-student:m001-mongodb-basics@sandbox.q8r0r.mongodb.net/hw --collection hwup --type jsonArray --file C:/Users/MR/Desktop/labData.json`
- mongoexport : Export JSON file
    - `mongoexport --uri mongodb+srv://m001-student:<PASSWORD>@sandbox.q8r0r.mongodb.net/<DATABASE> --collection <COLLECTION> --type <FILETYPE> --out <FILENAME>`
- mongorestore : Import BSON file
    - `mongorestore --uri "mongodb+srv://<your username>:<your password>@<your cluster>.mongodb.net/sample_supplies"  --drop dump`
- mongodump : Export BSON file
    - `mongodump --uri "mongodb+srv://<your username>:<your password>@<your cluster>.mongodb.net/sample_supplies"`

[How to get started with MongoDB in 10 minutes](https://www.freecodecamp.org/news/learn-mongodb-a4ce205e7739/)

- **How to run MongoDB after installation from Command Prompt (Not Atlas)**
    
    
    1- Download & install the msi (executable file), or download the zip & extract it.
    
    2- Make a folder called (data) in C:/ , & make inside it another folder called (db).
    
    3- Run in a command prompt the following command & keep it open : mongod
    
    4- Open another command prompt and run the command : `mongo` or `mongo "mongodb+srv://m001-student:m001-mongodb-basics@sandbox.q8r0r.mongodb.net"`
    
    5- Then do whatever you want.
    
- **How to import JSON file to MongoDB Database in Command Prompt**
    
    
    - first check mongoimport.exe file in your bin folder(C:\Program Files\MongoDB\Server\4.4\bin) if it is not then download mongodb database tools([https://www.mongodb.com/try/download/database-tools](https://www.mongodb.com/try/download/database-tools))
    - copy extracted(unzip) files(inside unzipped bin) to bin folder(C:\Program Files\MongoDB\Server\4.4\bin), or in the MongoDB unzipped folder in another drive (if not installed through the exe file) (which is in my case E:\MongoDB\bin)
    - In terminal, use the following command to import the books.json to the MongoDB database server:
    `mongoimport c:\data\books.json -d bookdb -c books --drop`
    
    (The location of the file to be imported can be anywhere, & while importing, the location to be specified. Another example (E:/ABC/testdata.json))
    
    In this command:
    
    - First, start with the `mongoimport` command.
    - Next, specify the path to the books.json data file. In this example, it is `c:\data\books.json`.
    - Third, use `-d bookdb` to specify the target database, which is bookdb in this example.
    - Fourth, use `-c books` to specify the target collection, which is books in this case.
    - Finally, use the `--drop` flag to drop the collection if it exists before importing the data
    
    The mongoimport command to be used alone & separately before `mongod` & `mongo` commands. After importing, we can access MongoDB databse server & work on the imported data.
    

"I will use BGDB as a database & Bigdata as a collection name in the following commands":

- `show dbs` : to show the list of databases.
- `use BGDB` : use the BGDP database.
- `show collections` : list the avialable collections
- `db.Bigdata.find({"state": "NY"})` : find all states that are named "NY".
- `it` : iterates to show the next group of documents.
    - **it :** iterates through a **Cursor**
    - **Cursor** :  A **Pointer** to a result set of a query
    - **Pointer** : A direct address of th memory location
- `db.Bigdata.find({"state": "NY"}).count()` : find how many documents match the query
- `db.Bigdata.find({"state": "NY"}).pretty()` : return the result in a formatted way for easy reading

- *When inserting a document via the Data Explorer in Atlas, the _id value is already generated as an **ObjectID** type, while the rest of the fields are not.*
- MongoDB generates a value, so that there is one just in case. You can definitely change the default value to a different value or data type, as long as they are unique to this document and not an array.

- **Creating a Collection:**  There are two ways:
    - One way is to insert data into the collection:
    `db.Bigdata.insert({"name": "john", "age" : 22, "location": "colombo"})`
    This is going to create your collection Bigdata even if the collection does not exist. Then it will insert a document with name and age.
    - Second Way:
        - Creating a Non-Capped Collection:
        `db.createCollection("myCollection")`
        - Creating a Capped Collection:
        `db.createCollection("mySecondCollection", {capped : true, size : 2, max : 2})`
        A “capped collection” has a maximum document count that prevents overflowing documents. In this example, I have enabled capping, by setting its value to `true`. The `size : 2` means a limit of two megabytes, and `max: 2` sets the maximum number of documents to two.
- **Insertion:**
    1. `insertOne()` is used to insert a single document only.
    `db.myCollection.insertOne({"name": "navindu", "age": 22})`
    2. `insertMany()` is used to insert more than one document.
    3. `insert()` is used to insert documents as many as you want.
    - **If the insertion is** **ordered**, if it encounters duplicate keys error (documents with the same `_id` key) then the insertion will stop. Even if the rest of the documents have unique IDs, they will not be inserted.
    `db.inspections.insert([{ "_id": 1, "test": 1 },{ "_id": 1, "test": 2 },{ "_id": 3, "test": 3 }])` : Here the insertion is ordered, so `{ "_id": 1, "test": 1 }` will be inserted, then a duplicate key error will rise when inserting        `{ "_id": 1, "test": 2 }` & the insertion operation will stop , so `{ "_id": 3, "test": 3 }` will not be inserted even if it has unique ID.
    - **If the insertion is** **not ordered**, every document that has unique ID (`_id` key) will be inserted ( for documents that have duplicate keys, only one will be inserted). This can be achieved using `{ "ordered": false }`
        
        `db.inspections.insert([{ "_id": 1, "test": 1 },{ "_id": 1, "test": 2 },{ "_id": 3, "test": 3 }],{ "ordered": false })` : Here the insertion is unordered, so `{ "_id": 3, "test": 3 }` will be inserted, & only one document with `"_id":1` will be inserted, resulting in 2 documents being inserted.
        
    - If the `_id` is not specified in the documents being inserted, all documents will be inserted because a unique `_id` with the value `ObjectId()` will be created for each one.
- **Update:**
    - `**updateOne` :** Update a **single** document in the zips collection where the zip field is equal to "12534", by setting the value of the "pop" field to 17630. 
    `db.zips.**updateOne**({ "zip": "12534" }, { "$set": { "pop": 17630 } })`
    - `**updateMany**` : Update **all** documents in the zips collection where the city field is equal to "HUDSON", by adding 10 to the current value of the "pop" field.
    `db.zips.**updateMany**({ "city": "HUDSON" }, { "$inc": { "pop": 10 } })`
    - `**update()**` : ***below command will update all documents with the specififed fields. Only when the "_id" is specified, the intended document will be updated.***   
    `db.hwuPeople.update({last_name : "burger"}, {$set: {title : "prof", role : "prof"}})`
    - **Adding an elemnt to an array**:
        - 1- Update one document in the grades collection where the student_id is 250, and the class_id field is 339 , by adding a document element to the "scores" array. "scores" is an array of documents [ { }, { }, { } .... ]
        `db.grades.updateOne({ "student_id": 250, "class_id": 339 }, { "$push": { "scores": { "type": "extra credit", "score": 100 }}})`
        - 2- Add a string element to the the "role" array. "role" is an array of strings [ "a", "b", "c" .... ]
        `db.hwuPeople.update({first_name: "manni"}, {$push: {role: "lab assistant"}})`
    - **If we try to update a field that doesnt exist, MongoDB will create it & update it with the given value.**
    - **If you try to update a document that is not there, nothing happens.**  However, mongoDB supports “**upserts**” (update or insert if there is no document found) : 
    `db.hwuPeople.update({first_name: "andy", role: "ra"}, {age: 47}, **{upsert:true}**)`
    This adds a new document, but it only contains “age” field, which is not the intended action !!  The right query is : 
    `db.hwuPeople.update({first_name:"andy",last_name:"proudlove",role:"ra"}, {first_name:"andy",last_name:"proudlove",role:"ra",age:47}, {upsert:true})`
    (Google "upsert" for more clarification)
    - **With a flexible schema you can add information to one document but not to others.** We can add additional fields to specific existing documents through update() : 
    `db.hwuPeople.update({last_name: "mcleod"}, {$set: {email:"kcm1@hw.ac.uk"}})`
- **Deletion:**
    - `**deleteOne**` : Delete **one** document that has test field equal to 3 : `db.inspections.**deleteOne**({ "test": 3 })`
    - `**deleteMany**` : Delete **all** the documents that have test field equal to 1 : `db.inspections.**deleteMany**({ "test": 1 })`
    - Remove a property (key) from a single document:
    `db.myCollection.update({name: "navindu"}, {**$unset**: age});`
    - `**remove()**` :
        - Remove all documents that match the specified criteria of age = 47
        `db.hwuPeople.remove({age: 47})`
        - Remove all documents in a collection : 
        `db.hwuPeople.remove({ })`
    - Drop the inspection collection with any indexes created for it : `db.inspection.drop()` . Or while importing we can use `--drop` to ensure the collection is dropped before insertion to prevent duplicate keys error.
    - **When all collections are dropped from a database, the database no longer appears in the list of databases when you run `show dbs`.**
    
- `**distinct()**` : find the unique values.
`db.hwuPeople.distinct("title")`
- `**{$exists: false}**` : to find a document which does not have a specific field.
`db.hwuPeople.find({age : {$exists: false}})` 
`db.hwup.find({"first_name" : "kareem", "age" : {$exists : false}})`
- **`findOne`** : to find ****only one document 
`db.listingsAndReviews.**findOne**({ },{ "address": 1, "_id": 0 })`

- **Comparison Operators**: Let’s say you want only to display people whose age is less than 25. You can use `$lt` to filter for this.
`db.myCollection.find({age : {$gt : 25, $lt : 100}})`
Similarly, `$gt` stands for "greater than", `$lte` is “less than or equal to”, `$gte` is “greater than or equal to” , `$eq` is "equal to" and `$ne` is “not equal”. They are used in 2 formats :
    - Applied on a field value. "It's used inside curly brackets". `{age : {$gt : 25}}`
    - Compare two field values. "It's using an array of values". `{ "$eq": [ "$end station id", "$start station id" ] }`

- **Counting**:
    - `length()` is for the length of documents, while `length` is for the length of a list.
    - `db.hwuPeople.**count**({age : 35})`   OR    `db.hwuPeople.find({age : 35}).**count()**`

- **Sorting & Limiting**: Sorting the result based on a specific field. Use 1 for ascending order, and -1 for descending order. Then limit the result to 10 documents only. In MongoDB, when using `sort()` & `limit()` together regardless of their order, it assumes we want to sort first them limit.
`db.hwuPeople.find({age : 35}).**sort({name: 1, age : -1}).limit(10)**`

- **Logical Operators:**
    - `$and`, `$or`, `$nor` (They include an array of statements): `{<operator> : [ {statement 1}, {statement 2}, ... ] }`
    **`db.routes.find({$and:[{$or :[ { "dst_airport": "KZN" }, { "src_airport": "KZN" }]},
                             {$or :[ { "airplane": "CR2" }, { "airplane": "A81" } ]}
                           ]}).pretty()`**
    - `$not` : `{$not : {statement}}`
    
- **Expressive Query Operator "`$expr`":**
    - Allows the use of aggregation expressions
    - Allows the use of variables, conditional statements (comparing field values)
        - `db.trips.find({ "$expr": { "$eq": [ "$end station id", "$start station id"] } }).`
        - `db.trips.find({ "$expr": { "$and": [ { "$gt": [ "$tripduration", 1200 ]},
                                 { "$eq": [ "$end station id", "$start station id" ]}
                               ]}}).count()` Look for trip durations that are greater than 1200 AND the "end station id" = "start station id"
    
- **The usage of the dollar sign "$":**
    - Denotes the use of an operator : like in `db.myCollection.find({age : {**$**gt : 25, **$**lt : 100}})`
    - Adresses the field value (Points at the value) : `db.trips.find({ "$expr": { "$eq": [ "**$**end station id", "**$**start station id"] } }).count()` Here we are comparing the equality of the "end station id" & "start station id" VALUES.
    
- **Array Operators:**
    - `{ <array field> : { **$size** : <number>} }` : return all documents where the specified array is exactly the given length.
    `{ "amenities": { $size : 20 } }`
    - `{ <array field> : { **$all** : <array of elements>} }` : return all documents where the specified array field contains all the given elements regardless of their order in the array.
    `{ "amenities": { $all : [""wifi, "shampoo", "parking"] } }`
    - Both `**$size**` & `**$all**` combined in a query : 
    `db.listingsAndReviews.find({ "amenities": { $size : 20, $all : [ "Internet", "Wifi",  "Kitchen" ] } } )`
    - If we try to query an array using a single element, the result will contain only documents where this array field contains this specific element. `db.listingsAndReviews.find({"amenities" : ["shampoo"] } )` Here we try to find documents where amenities have only shampoo as an element.
    - `{ <array field> :  { **$elemMatch** : { <field> : <value>} } }` : match al documents that contain an array field with at least one element that matches the specified criteria.
        - `db.grades.find( { "class_id": 431 }, { "scores": { **$elemMatch** : { "score": { "$gt": 85 } } } } )`
        Find me the documents with class id 431 & have scores array containing one element of "score" that has a value > 85.
        - `db.companies.find({ "relationships": { "**$elemMatch**": { "is_past": true, "person.first_name": "Mark" } } }, { "name": 1 })` Find documents where the "relationships" array contains "is_past" field in a sub-document of value "true" AND "first_name" field in the "person" document (which is a sub-document) of value "Mark".

- **Projection:**
To choose which fields in a document are returned : "0" for not displaying the field, "1" for displaying it. Dont combine 0s & 1s in a projection. The only combination allowed is to hide "_id" while displaying other fields, because "_id" will be displayed by default.
`db.hwuPeople.find({},{_id:0,first_name : 1, last_name : 1})`

- **Querying Sub-documents:
Dot-notation** is used to query nested elements in an array of sub-documents.
`db.<collection>.find( { "field1.otherField.alsoAfield" : <value>} )`
`db.trips.findOne({ "start station location.type": "Point" })`

- **Aggregation:**
As the order in the array matters, the aggregation framework is included in square brackets [just similar to arrays] to emphasize the importance of actions order, as the pipeline goes through the listed actions in order.
    - Using the aggregation framework find all documents that have Wifi as one of the amenities. Only include price and address in the resulting cursor. (This can achieved more concisely with `find()` )
    `db.listingsAndReviews.**aggregate**([ { "**$match**": { "amenities": "Wifi" } }, { "**$project**": { "price": 1, "address": 1, "_id": 0 }}])`
    - Project only the address field value for each document, then group all documents into one document per address.country value. `$project` is executed first, then `$group`
    `db.listingsAndReviews.**aggregate**([ { "**$project**": { "address": 1, "_id": 0 }}, { "**$group**": { "**_id**": "$address.country" }}])`
        - `"_id"` : the grouping by field
    - Project only the address field value for each document, then group all documents into one document per address.country value, and count one for each document in each group.
    `db.listingsAndReviews.**aggregate**([ { "**$project**": { "address": 1, "_id": 0 }}, { "**$group**": { "**_id**": "$address.country", "count": { "$sum": 1 } } } ])`
        - `"_id"` : the grouping by field
        - `"count"` : a count field to be presented in the result. It applies `$sum` to each document of the value 1 (counting the number of documents)
    - `**$size**` : Counts and returns the total number of items in an array.
    `{ $size: <expression> }` The argument for `$size` **must resolve to an array (must be an array)**. If the argument for `$size` is missing or does not resolve to an array, $size errors.
    - FULL EXAMPLE : 
    `db.hwuPeople.**aggregate**([
                            {**$group**: { _id: "$title", uniqueLastnames: {$addToSet: "$last_name"} } },
                            {**$project**: { _id: 1, groups: {$size: "$uniqueLastnames"} } },
                            {**$sort**: { groups: -1 } },
                            {**$limit**: 1}
                            ])`
