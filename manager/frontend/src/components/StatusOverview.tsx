import { useQuery } from '@tanstack/react-query'
import { CheckCircle, Clock, Loader2, AlertCircle, Search } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { episodesApi } from '@/lib/api'

interface StatusOverviewProps {
  onStatusClick: (status: string | null) => void
}

export function StatusOverview({ onStatusClick }: StatusOverviewProps) {
  const { data: stats } = useQuery({
    queryKey: ['episode-stats'],
    queryFn: episodesApi.stats,
    refetchInterval: 5000,
  })

  if (!stats) return null

  const items = [
    {
      label: 'Discovered',
      count: stats.discovered,
      icon: Search,
      color: 'text-blue-500',
      status: 'discovered',
    },
    {
      label: 'Queued',
      count: stats.queued,
      icon: Clock,
      color: 'text-yellow-500',
      status: 'queued',
    },
    {
      label: 'Processing',
      count: stats.processing,
      icon: Loader2,
      color: 'text-purple-500',
      status: 'processing',
    },
    {
      label: 'Cleaned',
      count: stats.cleaned,
      icon: CheckCircle,
      color: 'text-green-500',
      status: 'cleaned',
    },
    {
      label: 'Failed',
      count: stats.failed,
      icon: AlertCircle,
      color: 'text-red-500',
      status: 'failed',
    },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
      {items.map((item) => (
        <Card
          key={item.status}
          className="cursor-pointer hover:bg-accent transition-colors"
          onClick={() => onStatusClick(item.status)}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <item.icon className={`h-5 w-5 ${item.color}`} />
              <div>
                <div className="text-2xl font-bold">{item.count}</div>
                <div className="text-sm text-muted-foreground">{item.label}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
