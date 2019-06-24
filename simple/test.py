
class Book():
    def __call__(self, book_type):
        return book_type
    def printBook(self, book_type):
        print(book_type, 'from printBook')


Book().printBook('fire')
returned_book = Book()('fire')
