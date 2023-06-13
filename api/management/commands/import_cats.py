import csv
from django.core.management.base import BaseCommand
from ...models import CatBreed

class Command(BaseCommand):
    help = 'Batch import cats from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', help='Path to the CSV file')

    def handle(self, *args, **options):
        csv_file = options['csv_file']

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row

            for row in reader:
                id,name, cat_id = row  # Assuming the CSV has two columns: ID and Name

                # Create a new Cat object and save it to the database
                cat = CatBreed(id=int(cat_id), name=name)
                cat.save()

        