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
import { ListPlus, EyeOff, Eye, ChevronLeft, ChevronRight, X, Undo2, XCircle, AlertCircle, ArrowUp, ArrowDown, ArrowUpDown, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
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

type TabType = 'active' | 'queued' | 'cleaned' | 'ignored'

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
  const [errorModalOpen, setErrorModalOpen] = useState(false)
  const [selectedError, setSelectedError] = useState<{ title: string; error: string } | null>(null)
  const [sortBy, setSortBy] = useState<string>('published_at')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null)
  const [episodeModalOpen, setEpisodeModalOpen] = useState(false)

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
    queryKey: ['episodes', selectedFeedId, activeTab, statusFilter, sortBy, sortOrder, page],
    queryFn: () => {
      // If we have a specific status filter from StatusOverview, use that
      if (statusFilter) {
        return episodesApi.list({
          feed_id: selectedFeedId,
          status: statusFilter,
          sort_by: sortBy,
          sort_order: sortOrder,
          page,
          page_size: pageSize,
        })
      }

      // Otherwise, use tab logic
      if (activeTab === 'ignored') {
        return episodesApi.list({
          feed_id: selectedFeedId,
          status: 'ignored',
          sort_by: sortBy,
          sort_order: sortOrder,
          page,
          page_size: pageSize,
        })
      }

      if (activeTab === 'cleaned') {
        return episodesApi.list({
          feed_id: selectedFeedId,
          status: 'cleaned',
          sort_by: sortBy,
          sort_order: sortOrder,
          page,
          page_size: pageSize,
        })
      }

      if (activeTab === 'queued') {
        // Queued tab: show queued and processing episodes
        return episodesApi.list({
          feed_id: selectedFeedId,
          exclude_statuses: 'discovered,cleaned,ignored,failed',
          sort_by: sortBy,
          sort_order: sortOrder,
          page,
          page_size: pageSize,
        })
      }

      // Active tab: exclude ignored, cleaned, queued, and processing
      return episodesApi.list({
        feed_id: selectedFeedId,
        exclude_statuses: 'ignored,cleaned,queued,processing',
        sort_by: sortBy,
        sort_order: sortOrder,
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

  const reprocessMutation = useMutation({
    mutationFn: episodesApi.reprocessBulk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['episodes'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setRowSelection({})
    },
  })

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(column)
      setSortOrder('desc')
    }
    setPage(1)
  }

  const SortableHeader = ({ column, label }: { column: string; label: string }) => (
    <button
      onClick={() => handleSort(column)}
      className="flex items-center gap-1 hover:text-foreground transition-colors"
    >
      {label}
      {sortBy === column ? (
        sortOrder === 'asc' ? (
          <ArrowUp className="h-4 w-4" />
        ) : (
          <ArrowDown className="h-4 w-4" />
        )
      ) : (
        <ArrowUpDown className="h-4 w-4 opacity-50" />
      )}
    </button>
  )

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
        header: () => <SortableHeader column="title" label="Episode" />,
        cell: ({ row }) => (
          <div className="max-w-md truncate font-medium" title={row.original.title}>
            {row.original.title}
          </div>
        ),
      },
      {
        accessorKey: 'published_at',
        header: () => <SortableHeader column="published_at" label="Published" />,
        cell: ({ row }) => {
          const date = row.original.published_at
          if (!date) return <span className="text-muted-foreground">—</span>
          return format(new Date(date), 'MMM d, yyyy')
        },
      },
      {
        accessorKey: 'status',
        header: () => <SortableHeader column="status" label="Status" />,
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
          const title = row.original.title
          if (status !== 'failed' || !errorMessage) return null
          // Show first line of error, truncated
          const firstLine = errorMessage.split('\n')[0]
          return (
            <button
              onClick={() => {
                setSelectedError({ title, error: errorMessage })
                setErrorModalOpen(true)
              }}
              className="max-w-[300px] truncate text-sm text-destructive text-left hover:underline cursor-pointer"
            >
              {firstLine}
            </button>
          )
        },
      },
    ],
    [setSelectedError, setErrorModalOpen, sortBy, sortOrder]
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
        return status === 'cleaned' // Allow selecting cleaned episodes for reprocessing
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

  const handleReprocessSelected = () => {
    const ids = selectedEpisodes
      .filter((ep) => ep.status === 'cleaned')
      .map((ep) => ep.id)
    if (ids.length > 0) {
      reprocessMutation.mutate(ids)
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
            Inbox
          </Button>
          <Button
            variant={activeTab === 'queued' && !statusFilter ? 'default' : 'outline'}
            size="sm"
            onClick={() => handleTabChange('queued')}
          >
            <ListPlus className="mr-2 h-4 w-4" />
            Queued
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
            {isCleanedMode && selectedEpisodes.some((ep) => ep.status === 'cleaned') && (
              <Button
                onClick={handleReprocessSelected}
                disabled={reprocessMutation.isPending}
                size="sm"
              >
                <RotateCcw className="mr-2 h-4 w-4" />
                Reprocess ({selectedEpisodes.filter((ep) => ep.status === 'cleaned').length})
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
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={(e) => {
                    // Don't open modal if clicking on checkbox, button, or link
                    const target = e.target as HTMLElement
                    if (
                      target.closest('button') ||
                      target.closest('input') ||
                      target.closest('a')
                    ) {
                      return
                    }
                    setSelectedEpisode(row.original)
                    setEpisodeModalOpen(true)
                  }}
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
                  {activeTab === 'ignored' ? 'No ignored episodes.' : activeTab === 'cleaned' ? 'No cleaned episodes yet.' : activeTab === 'queued' ? 'No queued episodes.' : 'No episodes in inbox.'}
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

      {/* Error Details Modal */}
      <Dialog open={errorModalOpen} onOpenChange={setErrorModalOpen}>
        <DialogContent>
          <DialogHeader onClose={() => setErrorModalOpen(false)}>
            <DialogTitle>Error Details</DialogTitle>
          </DialogHeader>
          {selectedError && (
            <div className="space-y-4">
              <div>
                <span className="text-sm text-muted-foreground">Episode:</span>
                <p className="font-medium">{selectedError.title}</p>
              </div>
              <div>
                <span className="text-sm text-muted-foreground">Error:</span>
                <pre className="mt-2 p-4 bg-muted rounded-md text-sm overflow-auto max-h-[400px] whitespace-pre-wrap font-mono">
                  {selectedError.error}
                </pre>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Episode Details Modal */}
      <Dialog open={episodeModalOpen} onOpenChange={setEpisodeModalOpen}>
        <DialogContent>
          <DialogHeader onClose={() => setEpisodeModalOpen(false)}>
            <DialogTitle>Episode Details</DialogTitle>
          </DialogHeader>
          {selectedEpisode && (
            <div className="space-y-4">
              {/* Header with image */}
              <div className="flex gap-4">
                {selectedEpisode.image_url && (
                  <img
                    src={selectedEpisode.image_url}
                    alt=""
                    className="w-20 h-20 rounded object-cover flex-shrink-0"
                  />
                )}
                <div className="flex-1 min-w-0">
                  <p className="font-medium">{selectedEpisode.title}</p>
                  <p className="text-sm text-muted-foreground">{selectedEpisode.feed_title}</p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <span className="text-sm text-muted-foreground">Status</span>
                  <div className="mt-1">
                    <Badge variant={statusVariantMap[selectedEpisode.status]}>
                      {selectedEpisode.status.charAt(0).toUpperCase() + selectedEpisode.status.slice(1)}
                    </Badge>
                  </div>
                </div>

                <div>
                  <span className="text-sm text-muted-foreground">Published</span>
                  <p>
                    {selectedEpisode.published_at
                      ? format(new Date(selectedEpisode.published_at), 'MMM d, yyyy')
                      : '—'}
                  </p>
                </div>

                <div>
                  <span className="text-sm text-muted-foreground">Duration</span>
                  <p>
                    {selectedEpisode.duration
                      ? `${Math.floor(selectedEpisode.duration / 60)}m ${selectedEpisode.duration % 60}s`
                      : '—'}
                  </p>
                </div>
              </div>

              {selectedEpisode.description && (
                <div>
                  <span className="text-sm text-muted-foreground">Description</span>
                  <div
                    className="mt-1 text-sm max-h-[150px] overflow-auto prose prose-sm dark:prose-invert"
                    dangerouslySetInnerHTML={{ __html: selectedEpisode.description }}
                  />
                </div>
              )}

              <div>
                <span className="text-sm text-muted-foreground">GUID</span>
                <p className="text-sm font-mono break-all bg-muted p-2 rounded">{selectedEpisode.guid}</p>
              </div>

              <div>
                <span className="text-sm text-muted-foreground">Source Audio URL</span>
                <p className="text-sm font-mono break-all bg-muted p-2 rounded">
                  <a
                    href={selectedEpisode.audio_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    {selectedEpisode.audio_url}
                  </a>
                </p>
              </div>

              {selectedEpisode.local_filename && (
                <div>
                  <span className="text-sm text-muted-foreground">Cleaned File</span>
                  <p className="text-sm font-mono break-all bg-muted p-2 rounded">
                    {selectedEpisode.local_filename}
                  </p>
                </div>
              )}

              {selectedEpisode.error_message && (
                <div>
                  <span className="text-sm text-muted-foreground">Error</span>
                  <pre className="mt-1 p-3 bg-destructive/10 text-destructive rounded-md text-sm overflow-auto max-h-[200px] whitespace-pre-wrap font-mono">
                    {selectedEpisode.error_message}
                  </pre>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4 pt-2 border-t text-sm text-muted-foreground">
                <div>
                  <span>Created:</span>{' '}
                  {format(new Date(selectedEpisode.created_at), 'MMM d, yyyy h:mm a')}
                </div>
                <div>
                  <span>Updated:</span>{' '}
                  {format(new Date(selectedEpisode.updated_at), 'MMM d, yyyy h:mm a')}
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
