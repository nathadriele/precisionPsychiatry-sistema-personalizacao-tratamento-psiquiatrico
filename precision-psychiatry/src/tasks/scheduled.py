from celery.schedules import crontab
from src.tasks.celery_app import app
from src.tasks import tasks

app.conf.beat_schedule = {
    "cleanup-old-predictions": {
        "task": "src.tasks.tasks.cleanup_old_predictions",
        "schedule": crontab(hour=2, minute=0),  # 2 AM diariamente
        "args": (30,),
    },
    "generate-daily-report": {
        "task": "src.tasks.tasks.generate_daily_report",
        "schedule": crontab(hour=8, minute=0),  # 8 AM diariamente
    },
}