import { useEffect, useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
  type RowSelectionState,
} from '@tanstack/react-table'
import { format } from 'date-fns'
import { ListPlus, EyeOff, Eye, ChevronLeft, ChevronRight, X, Undo2, XCircle, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { episodesApi, feedsApi } from '@/lib/api'
import type { Episode, EpisodeStatus, Feed } from '@/types/api'

const statusVariantMap: Record<EpisodeStatus, 'default' | 'secondary' | 'destructive' | 'outline'> = {
  discovered: 'secondary',
  queued: 'outline',
  processing: 'default',
  cleaned: 'default',
  failed: 'destructive',
  ignored: 'secondary',
}

type TabType = 'active' | 'cleaned' | 'ignored'

interface EpisodeTableProps {
  initialStatusFilter?: string | null
  onClearFilter?: () => void
}

export function EpisodeTable({ initialStatusFilter, onClearFilter }: EpisodeTableProps) {
  const queryClient = useQueryClient()
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})
  const [activeTab, setActiveTab] = useState<TabType>('active')
  const [selectedFeedId, setSelectedFeedId] = useState<number | undefined>()
  const [statusFilter, setStatusFilter] = useState<string | undefined>(initialStatusFilter ?? undefined)
  const [page, setPage] = useState(1)
  const pageSize = 25

  // Update status filter when prop changes from parent (StatusOverview click)
  useEffect(() => {
    // Only sync when prop is a valid status string
    if (typeof initialStatusFilter === 'string' && initialStatusFilter.length > 0) {
      setStatusFilter(initialStatusFilter)
      setActiveTab('active') // Reset tab when using status filter
      setRowSelection({})
      setPage(1)
    }
  }, [initialStatusFilter])

  const { data, isLoading, error } = useQuery({
    queryKey: ['episodes', selectedFeedId, activeTab, statusFilter, page],
    queryFn: () => {
      // If we have a specific status filter from StatusOverview, use that
      if (statusFilter) {
        return episodesApi.list({
          feed_id: selectedFeedId,
          status: statusFilter,
          page,
          page_size: pageSize,
        })
      }

      // Otherwise, use tab logic
      if (activeTab === 'ignored') {
        return episodesApi.list({
          feed_id: selectedFeedId,
          status: 'ignored',
          page,
          page_size: pageSize,
        })
      }

      if (activeTab === 'cleaned') {
        return episodesApi.list({
          feed_id: selectedFeedId,
          status: 'cleaned',
          page,
          page_size: pageSize,
        })
      }

      // Active tab: exclude ignored and cleaned
      return episodesApi.list({
        feed_id: selectedFeedId,
        exclude_statuses: 'ignored,cleaned',
        page,
        page_size: pageSize,
      })
    },
  })


  const { data: feeds = [] } = useQuery({
    queryKey: ['feeds'],
    queryFn: feedsApi.list,
  })

  const episodes = useMemo(() => data?.items ?? [], [data?.items])
  const totalPages = data?.total_pages ?? 1
  const total = data?.total ?? 0

  const queueMutation = useMutation({
    mutationFn: episodesApi.queueBulk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['episodes'] })
      setRowSelection({})
    },
  })

  const ignoreMutation = useMutation({
    mutationFn: episodesApi.ignoreBulk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['episodes'] })
      setRowSelection({})
    },
  })

  const unignoreMutation = useMutation({
    mutationFn: episodesApi.unignoreBulk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['episodes'] })
      setRowSelection({})
    },
  })

  const unqueueMutation = useMutation({
    mutationFn: episodesApi.unqueueBulk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['episodes'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setRowSelection({})
    },
  })

  const failMutation = useMutation({
    mutationFn: episodesApi.failBulk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['episodes'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setRowSelection({})
    },
  })

  const columns = useMemo<ColumnDef<Episode>[]>(
    () => [
      {
        id: 'select',
        header: ({ table }) => (
          <Checkbox
            checked={
              table.getIsAllPageRowsSelected() ||
              (table.getIsSomePageRowsSelected() && 'indeterminate')
            }
            onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
            aria-label="Select all"
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            checked={row.getIsSelected()}
            onCheckedChange={(value) => row.toggleSelected(!!value)}
            disabled={!row.getCanSelect()}
            aria-label="Select row"
          />
        ),
        enableSorting: false,
      },
      {
        accessorKey: 'feed_title',
        header: 'Podcast',
        cell: ({ row }) => (
          <div className="max-w-[150px] truncate text-sm" title={row.original.feed_title}>
            {row.original.feed_title}
          </div>
        ),
      },
      {
        accessorKey: 'title',
        header: 'Episode',
        cell: ({ row }) => (
          <div className="max-w-md truncate font-medium" title={row.original.title}>
            {row.original.title}
          </div>
        ),
      },
      {
        accessorKey: 'published_at',
        header: 'Published',
        cell: ({ row }) => {
          const date = row.original.published_at
          if (!date) return <span className="text-muted-foreground">—</span>
          return format(new Date(date), 'MMM d, yyyy')
        },
      },
      {
        accessorKey: 'status',
        header: 'Status',
        cell: ({ row }) => {
          const status = row.original.status
          const errorMessage = row.original.error_message
          return (
            <div className="flex items-center gap-2">
              <Badge variant={statusVariantMap[status]}>
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </Badge>
              {status === 'failed' && errorMessage && (
                <span title={errorMessage} className="cursor-help">
                  <AlertCircle className="h-4 w-4 text-destructive" />
                </span>
              )}
            </div>
          )
        },
      },
      {
        id: 'error',
        header: 'Error',
        cell: ({ row }) => {
          const status = row.original.status
          const errorMessage = row.original.error_message
          if (status !== 'failed' || !errorMessage) return null
          // Show first line of error, truncated
          const firstLine = errorMessage.split('\n')[0]
          return (
            <div
              className="max-w-[300px] truncate text-sm text-destructive"
              title={errorMessage}
            >
              {firstLine}
            </div>
          )
        },
      },
    ],
    []
  )

  // Determine special modes based on tab or status filter
  const isIgnoredMode = activeTab === 'ignored' || statusFilter === 'ignored'
  const isCleanedMode = activeTab === 'cleaned' || statusFilter === 'cleaned'

  const coreRowModel = useMemo(() => getCoreRowModel(), [])

  const table = useReactTable({
    data: episodes,
    columns,
    state: {
      rowSelection,
    },
    enableRowSelection: (row) => {
      const status = row.original.status
      if (isCleanedMode) {
        return false // No bulk actions on cleaned episodes
      }
      if (isIgnoredMode) {
        return status === 'ignored'
      }
      // Allow selecting discovered, failed, queued, and processing episodes
      return status === 'discovered' || status === 'failed' || status === 'queued' || status === 'processing'
    },
    onRowSelectionChange: setRowSelection,
    getCoreRowModel: coreRowModel,
    getRowId: (row) => String(row.id),
  })

  const selectedEpisodes = table
    .getSelectedRowModel()
    .rows.map((row) => row.original)

  const handleQueueSelected = () => {
    const ids = selectedEpisodes.map((ep) => ep.id)
    queueMutation.mutate(ids)
  }

  const handleIgnoreSelected = () => {
    const ids = selectedEpisodes.map((ep) => ep.id)
    ignoreMutation.mutate(ids)
  }

  const handleUnignoreSelected = () => {
    const ids = selectedEpisodes.map((ep) => ep.id)
    unignoreMutation.mutate(ids)
  }

  const handleUnqueueSelected = () => {
    const ids = selectedEpisodes.filter((ep) => ep.status === 'queued').map((ep) => ep.id)
    if (ids.length > 0) {
      unqueueMutation.mutate(ids)
    }
  }

  const handleFailSelected = () => {
    const ids = selectedEpisodes
      .filter((ep) => ep.status === 'queued' || ep.status === 'processing')
      .map((ep) => ep.id)
    if (ids.length > 0) {
      failMutation.mutate(ids)
    }
  }

  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab)
    setStatusFilter(undefined) // Clear status filter when changing tabs
    if (onClearFilter) onClearFilter()
    setRowSelection({})
    setPage(1)
  }

  const handleClearStatusFilter = () => {
    setStatusFilter(undefined)
    if (onClearFilter) onClearFilter()
    setRowSelection({})
    setPage(1)
  }

  const handleFeedFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value
    setSelectedFeedId(value ? Number(value) : undefined)
    setRowSelection({})
    setPage(1)
  }

  if (error) {
    return (
      <div className="rounded-md border border-destructive">
        <div className="h-96 flex items-center justify-center text-destructive">
          Error loading episodes: {error.message}
        </div>
      </div>
    )
  }

  if (isLoading && !data) {
    return (
      <div className="rounded-md border">
        <div className="h-96 flex items-center justify-center text-muted-foreground">
          Loading episodes...
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Status Filter Badge */}
      {statusFilter && (
        <div className="flex items-center gap-2 p-2 bg-muted rounded-md">
          <span className="text-sm">
            Showing: <strong>{statusFilter.charAt(0).toUpperCase() + statusFilter.slice(1)}</strong> episodes
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClearStatusFilter}
            className="h-6 w-6 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Tabs and Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex gap-2">
          <Button
            variant={activeTab === 'active' && !statusFilter ? 'default' : 'outline'}
            size="sm"
            onClick={() => handleTabChange('active')}
          >
            Active
          </Button>
          <Button
            variant={activeTab === 'cleaned' && !statusFilter ? 'default' : 'outline'}
            size="sm"
            onClick={() => handleTabChange('cleaned')}
          >
            Cleaned
          </Button>
          <Button
            variant={activeTab === 'ignored' && !statusFilter ? 'default' : 'outline'}
            size="sm"
            onClick={() => handleTabChange('ignored')}
          >
            <EyeOff className="mr-2 h-4 w-4" />
            Ignored
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <label htmlFor="feed-filter" className="text-sm text-muted-foreground">
            Filter by podcast:
          </label>
          <select
            id="feed-filter"
            value={selectedFeedId ?? ''}
            onChange={handleFeedFilterChange}
            className="px-3 py-1 border rounded-md bg-background text-sm"
          >
            <option value="">All podcasts</option>
            {feeds.map((feed: Feed) => (
              <option key={feed.id} value={feed.id}>
                {feed.title}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Bulk Actions Toolbar */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {selectedEpisodes.length > 0 ? (
            <span>{selectedEpisodes.length} episode(s) selected</span>
          ) : (
            <span>{total} total episodes</span>
          )}
        </div>
        {selectedEpisodes.length > 0 && (
          <div className="flex gap-2">
            {!isIgnoredMode && (
              <>
                {/* Queue - for discovered/failed episodes */}
                {selectedEpisodes.some((ep) => ep.status === 'discovered' || ep.status === 'failed') && (
                  <Button
                    onClick={handleQueueSelected}
                    disabled={queueMutation.isPending}
                    size="sm"
                  >
                    <ListPlus className="mr-2 h-4 w-4" />
                    Queue
                  </Button>
                )}
                {/* Unqueue - for queued episodes */}
                {selectedEpisodes.some((ep) => ep.status === 'queued') && (
                  <Button
                    onClick={handleUnqueueSelected}
                    disabled={unqueueMutation.isPending}
                    size="sm"
                    variant="outline"
                  >
                    <Undo2 className="mr-2 h-4 w-4" />
                    Unqueue
                  </Button>
                )}
                {/* Mark Failed - for queued/processing episodes */}
                {selectedEpisodes.some((ep) => ep.status === 'queued' || ep.status === 'processing') && (
                  <Button
                    onClick={handleFailSelected}
                    disabled={failMutation.isPending}
                    size="sm"
                    variant="destructive"
                  >
                    <XCircle className="mr-2 h-4 w-4" />
                    Mark Failed
                  </Button>
                )}
                {/* Ignore - for discovered/failed episodes */}
                {selectedEpisodes.some((ep) => ep.status === 'discovered' || ep.status === 'failed') && (
                  <Button
                    onClick={handleIgnoreSelected}
                    disabled={ignoreMutation.isPending}
                    size="sm"
                    variant="outline"
                  >
                    <EyeOff className="mr-2 h-4 w-4" />
                    Ignore
                  </Button>
                )}
              </>
            )}
            {isIgnoredMode && (
              <Button
                onClick={handleUnignoreSelected}
                disabled={unignoreMutation.isPending}
                size="sm"
              >
                <Eye className="mr-2 h-4 w-4" />
                Restore ({selectedEpisodes.length})
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Data Table */}
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={row.getIsSelected() && 'selected'}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  {activeTab === 'ignored' ? 'No ignored episodes.' : activeTab === 'cleaned' ? 'No cleaned episodes yet.' : 'No episodes found.'}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Page {page} of {totalPages}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
