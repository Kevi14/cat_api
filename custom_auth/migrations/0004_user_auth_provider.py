# Generated by Django 4.2.1 on 2023-06-17 12:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('custom_auth', '0003_user_is_admin'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='auth_provider',
            field=models.CharField(blank=True, max_length=255, null=True, verbose_name='auth provider'),
        ),
    ]
